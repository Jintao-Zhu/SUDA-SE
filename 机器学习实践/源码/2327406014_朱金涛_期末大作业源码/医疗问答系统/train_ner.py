import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from torch.optim import AdamW
from seqeval.metrics import f1_score, classification_report
import numpy as np

# ================= 配置区域 =================
BATCH_SIZE = 32      # 稍微调大一点Batch Size加快速度
EPOCHS = 3           # 训练3轮
MAX_LEN = 128        # 最大长度
LR = 3e-5            # 学习率
MODEL_NAME = 'bert-base-chinese'
FILE_PATH = 'D:\\课程资料\\机器学习实践\\期末大作业\\期末大作业\\医疗问答\\ner_data_aug.txt'
# ===========================================

# 1. 读取数据
def load_data(file_path):
    sentences = []
    labels = []
    current_sent = []
    current_label = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sent:
                    sentences.append(current_sent)
                    labels.append(current_label)
                    current_sent = []
                    current_label = []
                continue
            parts = line.split()
            if len(parts) == 2:
                word, label = parts
                current_sent.append(word)
                current_label.append(label)
    if current_sent:
        sentences.append(current_sent)
        labels.append(current_label)
    return sentences, labels

print("正在读取数据...")
sentences, labels = load_data(FILE_PATH)
print(f"共读取到 {len(sentences)} 个句子。")

# 2. 标签映射
unique_labels = sorted(list(set(l for sent in labels for l in sent)))
tag2id = {tag: i for i, tag in enumerate(unique_labels)}
id2tag = {i: tag for tag, i in tag2id.items()}
print(f"标签列表: {unique_labels}")

# 3. 改进版数据集类 (核心修复部分)
class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, tag2id, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sent = self.sentences[idx]
        label = self.labels[idx]
        
        # 逐字构建，保证对齐
        input_ids = [self.tokenizer.cls_token_id] # [CLS]
        label_ids = [-100]                        # [CLS] 对应的标签忽略
        
        for w, l in zip(sent, label):
            # 对每个字分词 (防止 1个字变成多个token)
            word_tokens = self.tokenizer.tokenize(w)
            if not word_tokens:
                continue
                
            token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            input_ids.extend(token_ids)
            
            # 标签对齐：第一个sub-token给真实标签，其他的给-100
            label_ids.append(self.tag2id[l])
            if len(token_ids) > 1:
                label_ids.extend([-100] * (len(token_ids) - 1))
        
        # 长度截断 (保留最后的 [SEP] 空间)
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[:self.max_len - 1]
            label_ids = label_ids[:self.max_len - 1]
            
        # 添加 [SEP]
        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(-100)
        
        # 填充 (Padding)
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            label_ids += [-100] * padding_len
            mask = [1] * (self.max_len - padding_len) + [0] * padding_len
        else:
            mask = [1] * self.max_len
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

# 4. 初始化
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
dataset = NERDataset(sentences, labels, tokenizer, tag2id, MAX_LEN)

# 划分数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag2id))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)

# 5. 训练
print("开始训练...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if step % 50 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
            
    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

# 6. 评估
print("开始评估...")
model.eval()
true_labels, pred_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask=mask).logits
        preds = torch.argmax(logits, dim=2).cpu().numpy()
        labels = labels.cpu().numpy()
        
        for i in range(len(labels)):
            t_true, t_pred = [], []
            for j in range(len(labels[i])):
                if labels[i][j] != -100:
                    t_true.append(id2tag[labels[i][j]])
                    t_pred.append(id2tag[preds[i][j]])
            true_labels.append(t_true)
            pred_labels.append(t_pred)

print(f"\n整体 F1 Score: {f1_score(true_labels, pred_labels):.4f}")
print(classification_report(true_labels, pred_labels))

# 7. 保存
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
print("模型已保存！")