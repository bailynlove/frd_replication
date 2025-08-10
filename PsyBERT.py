import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse

warnings.filterwarnings("ignore")

# --- 命令行参数 ---
parser = argparse.ArgumentParser(description='Train Transformer Model for Fake Review Detection')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training.')
args = parser.parse_args()

# --- 配置 ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_model": "bert-base-uncased",
    "learning_rate": 2e-5,  # 经典的微调学习率
    "num_classes": 2,
    "best_model_path": "bert_fake_review_detector.pth"
}

print(f"使用设备: {CONFIG['device']}")


# --- 数据集类 ---
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

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


# --- Transformer 分类模型 ---
class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # 增加一个 dropout 层以防止过拟合
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] token 的输出来进行分类
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


# --- 主执行块 ---
if __name__ == '__main__':
    # 1. 加载和预处理数据
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content'], inplace=True)

    # 标签映射: 1 (真评论) -> 1, 0 (假评论) -> 0
    df['label'] = df['is_recommended']

    # 2. 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")

    # 3. 初始化 Tokenizer 和数据集/数据加载器
    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    train_dataset = ReviewDataset(train_df, tokenizer)
    val_dataset = ReviewDataset(val_df, tokenizer)
    test_dataset = ReviewDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)

    # 4. 初始化模型、优化器、损失函数和学习率调度器
    model = BERTClassifier(CONFIG["bert_model"], num_classes=CONFIG["num_classes"]).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    # 5. 训练和验证循环
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
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

            progress_bar.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(CONFIG["device"])
                attention_mask = batch['attention_mask'].to(CONFIG["device"])
                labels = batch['label'].to(CONFIG["device"])
                outputs = model(input_ids, attention_mask)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        avg_train_loss = total_loss / len(train_loader)
        print(f"训练损失: {avg_train_loss:.4f} | 验证准确率: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])
            print(f"发现新的最佳模型，已保存至 {CONFIG['best_model_path']}")

    # 6. 在测试集上评估最佳模型
    print("\n--- 开始在测试集上进行最终评估 ---")
    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(CONFIG["device"])
            attention_mask = batch['attention_mask'].to(CONFIG["device"])
            labels = batch['label']
            outputs = model(input_ids, attention_mask)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)

    print("\n--- 测试结果 ---")
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")
    print("--------------------")