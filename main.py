import os
import sys
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt

# ======= 설정 ========
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-5
MODEL_PATH = "kcelectra_finetuned.pth"


test_sentences = [
    # 우파 - 정상
    "보수는 국가 안보와 자유를 중시하는 가치다.",
    "박정희 대통령 시절의 산업화는 역사적으로 평가받아야 한다.",
    "윤석열 정부의 대북 정책은 단호해서 좋다.",
    "시장 자율을 중시하는 경제정책이 더 적합하다고 본다.",
    "자유는 복지보다 더 중요한 가치일 수 있다.",
    "진보와 보수는 균형을 이루어야 한다.",
    "검찰의 독립성은 민주주의 핵심이다.",
    "민간 주도의 성장 전략이 효과적일 수 있다.",
    "한미동맹은 대한민국의 안보에 필수적이다.",
    "보수의 도덕성과 책임감도 중요한 가치다.",

    # 우파 - 애매 (경고)
    "문재인 정권은 나라를 망하게 만든 집단이다.",
    "좌파 새끼들은 늘 피해자 코스프레만 한다.",
    "이재명은 범죄자 같은 이미지가 있다.",
    "진보 진영은 항상 감성팔이만 한다.",
    "민주당은 퍼주기만 하는 집단이다.",
    "운동권 출신들이 나라 운영할 수준이 아니다.",
    "좌파는 남의 주머니서 돈 빼간다 생각한다.",
    "조국 사태 때 보면 진보는 위선자들이다.",
    "이 나라를 지켜낸 건 보수가 아니라면 누가 있냐?",
    "좌파 정권은 항상 북한에 굽신거린다.",

    # 우파 - 혐오 (차단)
    "문재인 저 빨갱이 새끼가 나라 다 말아먹었다.",
    "이재명은 감옥에서 썩어야 할 놈이다.",
    "좌빨들은 전부 김정은 개새끼들이다.",
    "진보 새끼들은 다 종북 간첩이다.",
    "촛불 시위한 놈들 전부 반역자다.",
    "페미충, 좌좀 다 꺼져야 나라가 산다.",
    "민주당은 범죄자 집단 그 자체다.",
    "전라도 좌파들은 사람 취급할 필요 없다.",
    "노무현은 자살해서 다행이다.",
    "좌파놈들은 나라에서 다 쫓아내야 한다."
]



def label_meaning(label):
    if label <= 1:
        return "❌ 비속어 - 매우 심함"
    elif label <= 3:
        return "⚠️ 비속어 또는 논란 가능"
    else:
        return "✅ 정상어"

# ======= 디바이스 자동 선택 ========
def get_device():
    if torch.cuda.is_available():
        print("✅ GPU 사용:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("⚠️ GPU 없음 → CPU 사용")
        return torch.device("cpu")

DEVICE = get_device()

# ======= 시드 고정 ========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ======= 데이터 로드 및 전처리 ========
train_df = pd.read_csv('./newhas_train.tsv', sep='\t')
valid_df = pd.read_csv('./newhas_valid.tsv', sep='\t')
test_df = pd.read_csv('./newhas_test.tsv', sep='\t')

train_df['label'] = train_df['label'].apply(lambda x: int(str(x).split(',')[0]))
valid_df['label'] = valid_df['label'].apply(lambda x: int(str(x).split(',')[0]))
test_df['label'] = test_df['label'].apply(lambda x: int(str(x).split(',')[0]))

# ======= 토크나이저 및 모델 불러오기 ========
MODEL_NAME = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
NUM_LABELS = len(train_df['label'].unique())
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)

# ======= 토큰화 및 데이터셋 구성 ========
def encode_data(sentences, labels):
    encodings = tokenizer(list(sentences), padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))

train_data = encode_data(train_df['document'], train_df['label'])
valid_data = encode_data(valid_df['document'], valid_df['label'])
test_data  = encode_data(test_df['document'],  test_df['label'])

train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=BATCH_SIZE)

# ======= 옵티마이저 & 스케줄러 ========
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

# ======= 평가 함수 ========
def evaluate():
    model.eval()
    preds, targets = [], []
    for batch in valid_loader:
        b_input_ids, b_input_mask, b_labels = [x.to(DEVICE) for x in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)
        preds.extend(pred.cpu().numpy())
        targets.extend(b_labels.cpu().numpy())
    print("✅ Validation Accuracy:", accuracy_score(targets, preds))
    print("✅ Validation F1 Score:", f1_score(targets, preds, average='macro'))
    print(classification_report(targets, preds))

# ======= 학습 함수 ========
def train():
    if os.path.exists(MODEL_PATH):
        print(f"📂 기존 모델 불러오기: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        print("🔁 이어서 학습 시작")
    else:
        print("🚀 새로운 모델로 학습 시작")

    try:
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

            print(f"📉 Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
            evaluate()

            print(f"\n🧪 [테스트 문장 예측] - Epoch {epoch+1}")
            for sentence in test_sentences:
                label = predict(sentence)
                meaning = label_meaning(label)
                print(f"[{sentence}] → 예측: {label} ({meaning})")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ 모델 저장 완료: {MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n⛔ 학습 중단됨: 키보드 인터럽트")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 현재까지 저장 완료: {MODEL_PATH}")


# ======= 예측 함수 ========
def predict(sentence):
    model.eval()
    encodings = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
    input_id = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_id, attention_mask=attention_mask)
    logits = outputs.logits
    return torch.argmax(logits, dim=1).item()

# ======= GUI ========
class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧼 욕설 필터링 채팅")
        self.setGeometry(100, 100, 500, 500)
        self.awaiting_confirmation = False
        self.last_input = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red; font-weight: bold;")

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("메시지를 입력하세요...")
        self.input_field.returnPressed.connect(self.handle_send)  # ⏎ 엔터 처리
        self.send_button = QPushButton("전송")
        self.send_button.clicked.connect(self.handle_send)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        layout.addWidget(self.chat_area)
        layout.addWidget(self.warning_label)
        layout.addLayout(input_layout)
        self.setLayout(layout)

    def handle_send(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return

        label = predict(user_input)

        # 1회 경고 조건
        if 2 <= label <= 4:
            if not self.awaiting_confirmation or self.last_input != user_input:
                self.warning_label.setText("⚠️ 이 메시지는 누군가를 비방할 수 있어요. 다시 한 번 눌러야 전송됩니다.")
                self.awaiting_confirmation = True
                self.last_input = user_input
                return

        # 0~1 (욕설 강함): 차단
        if label <= 1:
            self.warning_label.setText("❌ 비속어가 심하게 감지되어 메시지가 전송되지 않았습니다.")
            self.awaiting_confirmation = False
            return

        # 정상 전송
        self.warning_label.setText("")
        self.chat_area.append(f"👤 사용자: {user_input}")
        self.input_field.clear()
        self.awaiting_confirmation = False
        self.last_input = ""


# ======= 실행 ========
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("✅ 저장된 모델 로드 완료")
    
    t = input("학습을 시작하시겠습니까? (y/n): ").strip().lower()
    if t == 'y':
        train()
    else:
        print("학습을 건너뜁니다.")
        
    t = input("테스트 문장을 예측하시겠습니까? (y/n): ").strip().lower()
    
    if t == 'y': 
        print(f"\n🧪 [테스트 문장 예측]")
        for sentence in test_sentences:
            label = predict(sentence)
            meaning = label_meaning(label)
            print(f"[{sentence}] → 예측: {label} ({meaning})")
    else:
        print("테스트 문장 예측을 건너뜁니다.")

    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())
