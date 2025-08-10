import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import liwc  # 导入LIWC库
import xgboost as xgb  # 导入XGBoost

warnings.filterwarnings("ignore")

# --- 命令行参数 ---
parser = argparse.ArgumentParser(description='Train Psycholinguistics and Transformer Models')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for Transformer training.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for Transformer training.')
args = parser.parse_args()

# --- 配置 ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_model": "bert-base-uncased",
    "learning_rate": 2e-5,
    "best_model_path": "psybert_best_model.pth"
}

print(f"使用设备: {CONFIG['device']}")


# --- 流程A: 心理语言学特征 + 传统机器学习 ---

def train_with_psycholinguistics(train_df, test_df):
    print("\n--- 开始流程 A: 心理语言学特征 + XGBoost ---")

    # 1. 加载LIWC词典
    # liwc.load_token_parser需要一个词典文件的路径。
    # 您需要从网上下载一个LIWC的词典文件（通常是.dic格式），例如LIWC2007或LIWC2015的开源版本。
    # 这里假设您下载了名为 'LIWC2007_English.dic' 的文件并放在了脚本同目录下。
    try:
        parse, category_names = liwc.load_token_parser('LIWC2007_English.dic')
    except FileNotFoundError:
        print("\n错误: LIWC词典文件 'LIWC2007_English.dic' 未找到。")
        print("请从网上下载LIWC词典文件，并将其放在此脚本所在的目录。")
        print("流程 A 将被跳过。\n")
        return

    # 2. 为训练集和测试集提取LIWC特征
    print("正在为数据集提取LIWC特征...")
    X_train = []
    for text in tqdm(train_df['content'], desc="处理训练集"):
        tokens = text.lower().split()
        counts = Counter(category for token in tokens for category in parse(token))
        X_train.append([counts.get(cat, 0) for cat in category_names])

    X_test = []
    for text in tqdm(test_df['content'], desc="处理测试集"):
        tokens = text.lower().split()
        counts = Counter(category for token in tokens for category in parse(token))
        X_test.append([counts.get(cat, 0) for cat in category_names])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # 3. 训练XGBoost分类器
    print("正在训练XGBoost模型...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # 4. 评估模型
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')

    print("\n--- 流程 A 结果 ---")
    print(f"模型: XGBoost on LIWC Features")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("---------------------\n")


# --- 流程B: Transformer模型微调 ---

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['content'])
        label = int(row['label'])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


def train_transformer(train_df, val_df, test_df):
    print("\n--- 开始流程 B: Transformer (BERT) 微调 ---")

    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    train_dataset = ReviewDataset(train_df, tokenizer)
    val_dataset = ReviewDataset(val_df, tokenizer)
    test_dataset = ReviewDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = BERTClassifier(CONFIG["bert_model"], num_classes=2).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(CONFIG["device"])
            attention_mask = batch['attention_mask'].to(CONFIG["device"])
            labels = batch['label'].to(CONFIG["device"])

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(CONFIG["device"])
                attention_mask = batch['attention_mask'].to(CONFIG["device"])
                labels = batch['label'].to(CONFIG["device"])
                outputs = model(input_ids, attention_mask)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch + 1}: Train Loss={(total_loss / len(train_loader)):.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])

    print("\nTesting with best model...")
    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(CONFIG["device"])
            attention_mask = batch['attention_mask'].to(CONFIG["device"])
            labels = batch['label']
            outputs = model(input_ids, attention_mask)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print("\n--- 流程 B 结果 ---")
    print(f"模型: Fine-tuned BERT")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("---------------------\n")


# --- 主执行块 ---
if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content'], inplace=True)

    # 标签映射: 1 (真评论) -> 1, 0 (假评论) -> 0
    df['label'] = df['is_recommended']

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 运行流程A
    from collections import Counter

    train_with_psycholinguistics(train_df, test_df)

    # 运行流程B
    train_transformer(train_df, val_df, test_df)