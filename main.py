import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraModel
from torch.optim import AdamW
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 설정
PRETRAINED_MODEL_NAME = "monologg/koelectra-base-discriminator"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 클래스
class HateSpeechDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.data = pd.read_csv(filepath, sep='\t')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['comments'])
        label = int(self.data.iloc[idx]['label'])

        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 모델 정의
class ElectraClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes=2):
        super(ElectraClassifier, self).__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.electra.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)

# 데이터 로딩
tokenizer = ElectraTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
train_dataset = HateSpeechDataset("newhas_train.tsv", tokenizer, MAX_LEN)
valid_dataset = HateSpeechDataset("newhas_valid.tsv", tokenizer, MAX_LEN)
test_dataset  = HateSpeechDataset("newhas_test.tsv", tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 모델 준비
model = ElectraClassifier(PRETRAINED_MODEL_NAME).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# 학습 함수
def train(model, data_loader):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

# 평가 함수
def evaluate(model, data_loader):
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    return acc, f1

# 전체 학습 루프
for epoch in range(EPOCHS):
    print(f"\n[Epoch {epoch+1}]")
    train_loss = train(model, train_loader)
    val_acc, val_f1 = evaluate(model, valid_loader)
    print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

# 테스트 결과
test_acc, test_f1 = evaluate(model, test_loader)
print(f"\n[Test Performance] Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
