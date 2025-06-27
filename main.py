import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import Dataset, DatasetDict

# ======= 1. 설정 ========
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= 2. 시드 고정 ========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ======= 3. 데이터 불러오기 및 전처리 ========
train_df = pd.read_csv('./newhas_train.tsv', sep='\t')
valid_df = pd.read_csv('./newhas_valid.tsv', sep='\t')
test_df = pd.read_csv('./newhas_test.tsv', sep='\t')

# [중요] 라벨 타입 변환
# 👇 '2,3' 같은 라벨 처리
train_df['label'] = train_df['label'].apply(lambda x: int(str(x).split(',')[0]))
valid_df['label'] = valid_df['label'].apply(lambda x: int(str(x).split(',')[0]))
test_df['label'] = test_df['label'].apply(lambda x: int(str(x).split(',')[0]))


# ======= 4. 토크나이저 ========
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

def preprocess_sentences(sentences):
    return ['[CLS] ' + str(s) + ' [SEP]' for s in sentences]

def tokenize(sentences):
    tokenized = [tokenizer.encode(s, add_special_tokens=True, max_length=MAX_LEN, truncation=True) for s in sentences]
    padded = pad_sequences(tokenized, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    return torch.tensor(padded)

def create_masks(input_ids):
    return torch.tensor([[float(token > 0) for token in seq] for seq in input_ids])

# ======= 5. 입력 데이터 처리 ========
train_sentences = preprocess_sentences(train_df['document'])
valid_sentences = preprocess_sentences(valid_df['document'])
test_sentences = preprocess_sentences(test_df['document'])

train_inputs = tokenize(train_sentences)
valid_inputs = tokenize(valid_sentences)
test_inputs = tokenize(test_sentences)

train_masks = create_masks(train_inputs)
valid_masks = create_masks(valid_inputs)
test_masks = create_masks(test_inputs)

train_labels = torch.tensor(train_df['label'].values)
valid_labels = torch.tensor(valid_df['label'].values)
test_labels = torch.tensor(test_df['label'].values)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
test_data = TensorDataset(test_inputs, test_masks, test_labels)

train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=BATCH_SIZE)

# ======= 6. 모델 구성 ========
model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=len(train_df['label'].unique()))
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)

# ======= 7. 학습 및 평가 ========
def train():
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            b_input_ids, b_input_mask, b_labels = [x.to(DEVICE) for x in batch]
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        evaluate()

def evaluate():
    model.eval()
    predictions, true_labels = [], []
    for batch in valid_loader:
        b_input_ids, b_input_mask, b_labels = [x.to(DEVICE) for x in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())
    print("Validation Accuracy:", accuracy_score(true_labels, predictions))
    print("Validation F1 Score:", f1_score(true_labels, predictions, average='macro'))

# ======= 8. 예측 함수 ========
def predict(sentence):
    model.eval()
    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.encode(marked_text, add_special_tokens=True, max_length=MAX_LEN, truncation=True)
    input_id = pad_sequences([tokenized_text], maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    input_id = torch.tensor(input_id).to(DEVICE)
    attention_mask = torch.tensor([[float(i > 0) for i in input_id[0]]]).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_id, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label

# ======= 9. 실행 ========
if __name__ == "__main__":
    train()

    # 예측 테스트
    print("\n🔥 예측 테스트:")
    test_sentences = [
        "노무현 대통령님 사랑합니다",
        "노무현 운지 응디응디 딱",
        "노무현은 빨갱이",
        "노무현은 개새끼",
        "전라도 좌빨 종북좌파 깜둥이 개새끼들"
    ]

    for sent in test_sentences:
        label = predict(sent)
        print(f"[{sent}] → 예측 라벨: {label}")
