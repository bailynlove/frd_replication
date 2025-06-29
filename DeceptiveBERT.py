import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np

# --- 1. 配置参数 ---
# 在这里可以方便地修改所有超参数
class Config:
    CSV_PATH = '../spams_dataset/LA/outputs/full_data_0617.csv' # 你的数据集路径
    TEXT_COLUMN = 'content'
    LABEL_COLUMN = 'is_recommended'
    
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_LEN = 200  # 论文中使用的最大长度
    BATCH_SIZE = 16 # 根据你的显存调整
    EPOCHS = 3     # 论文中使用的Epochs
    LEARNING_RATE = 1e-3 # 论文中提到的学习率，因为只训练分类头，所以可以高一些
    
    TRAIN_SIZE = 0.7
    VALID_SIZE = 0.15
    TEST_SIZE = 0.15

# --- 2. 检查并设置设备 (支持 MPS) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ MPS is available! Using MPS as the device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ CUDA is available! Using CUDA as the device.")
else:
    device = torch.device("cpu")
    print("⚠️ MPS and CUDA not available, using CPU.")


# --- 3. 自定义数据集类 ---
class DeceptiveReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 4. 定义 DeceptiveBERT 模型 ---
# 完全按照论文 Figure 2 的结构复现
class DeceptiveBERT(nn.Module):
    def __init__(self, n_classes):
        super(DeceptiveBERT, self).__init__()
        self.bert = BertModel.from_pretrained(Config.PRE_TRAINED_MODEL_NAME)
        
        # 冻结BERT的所有参数，使其只作为特征提取器
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # 自定义的分类器，与论文结构一致
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.bert.config.hidden_size, 512), # 768 -> 512
            nn.ReLU(),
            nn.Linear(512, n_classes), # 512 -> 2
            nn.LogSoftmax(dim=1) # 论文中提到使用LogSoftmax
        )

    def forward(self, input_ids, attention_mask):
        # 我们只需要[CLS] token的输出用于分类
        # BERT模型的pooler_output就是[CLS] token经过一个全连接层和Tanh激活后的结果
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        
        # 将pooled_output送入分类器
        return self.classifier(pooled_output)

# --- 5. 训练和评估函数 ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    
    pbar = tqdm(data_loader, desc="Training", total=len(data_loader))
    for d in pbar:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': np.mean(losses)})

    return np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    
    # 用于存储所有预测和真实标签
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", total=len(data_loader))
        for d in pbar:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # LogSoftmax的输出是log-probabilities, 使用torch.exp转回概率，再用argmax取最大概率的索引
            _, preds = torch.max(outputs, dim=1)
            
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1, np.mean(losses)

# --- 主执行流程 ---
if __name__ == '__main__':
    # 加载数据
    print(f"Loading data from {Config.CSV_PATH}...")
    df = pd.read_csv(Config.CSV_PATH)
    
    # 选择所需列并处理缺失值
    df = df[[Config.TEXT_COLUMN, Config.LABEL_COLUMN]].dropna()
    df[Config.LABEL_COLUMN] = df[Config.LABEL_COLUMN].astype(int)
    
    # 检查标签分布
    print("\n--- Dataset Info ---")
    print("Label distribution:")
    print(df[Config.LABEL_COLUMN].value_counts(normalize=True))
    label_distribution = df[Config.LABEL_COLUMN].value_counts()
    
    # 如果数据集过大，可以抽样一部分进行快速实验
    # df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.PRE_TRAINED_MODEL_NAME)
    
    # 划分数据集 (70% 训练, 15% 验证, 15% 测试)
    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - Config.TRAIN_SIZE),
        random_state=42,
        stratify=df[Config.LABEL_COLUMN] # 确保各数据集标签分布一致
    )
    
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(Config.TEST_SIZE / (Config.VALID_SIZE + Config.TEST_SIZE)),
        random_state=42,
        stratify=df_temp[Config.LABEL_COLUMN]
    )

    print("\nDataset sizes:")
    print(f"Train: {len(df_train)}")
    print(f"Validation: {len(df_val)}")
    print(f"Test: {len(df_test)}")

    # 创建 DataLoader
    train_dataset = DeceptiveReviewDataset(
        texts=df_train[Config.TEXT_COLUMN].to_numpy(),
        labels=df_train[Config.LABEL_COLUMN].to_numpy(),
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN
    )
    train_data_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    val_dataset = DeceptiveReviewDataset(
        texts=df_val[Config.TEXT_COLUMN].to_numpy(),
        labels=df_val[Config.LABEL_COLUMN].to_numpy(),
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN
    )
    val_data_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    test_dataset = DeceptiveReviewDataset(
        texts=df_test[Config.TEXT_COLUMN].to_numpy(),
        labels=df_test[Config.LABEL_COLUMN].to_numpy(),
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN
    )
    test_data_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # 初始化模型、损失函数和优化器
    model = DeceptiveBERT(n_classes=len(label_distribution)).to(device)
    
    # 由于模型最后一层是LogSoftmax，所以使用NLLLoss（负对数似然损失）
    loss_fn = nn.NLLLoss().to(device)
    
    # 只优化分类器头的参数
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=Config.LEARNING_RATE)
    
    # 训练循环
    print("\n--- Starting Training ---")
    best_f1 = 0
    for epoch in range(Config.EPOCHS):
        print(f'Epoch {epoch + 1}/{Config.EPOCHS}')
        
        train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(df_train)
        )
        
        print(f'Train loss {train_loss}')

        val_acc, val_f1, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        
        print(f'Validation loss {val_loss}, Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}')

        # 保存F1-score最高的模型
        if val_f1 > best_f1:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_f1 = val_f1
            print("✨ New best model saved!")
    
    # 在测试集上评估最终模型
    print("\n--- Testing on the final model ---")
    # 加载表现最好的模型权重
    model.load_state_dict(torch.load('best_model_state.bin'))
    
    test_acc, test_pre, test_rec, test_f1, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
    )
    
    print("\n--- Final Baseline Results ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Pre: {test_pre:.4f}")
    print(f"Test Rec: {test_rec:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("--------------------------------")