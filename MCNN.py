# -*- coding: utf-8 -*-
"""
Strict Replication of MCNN with Multi-task Learning Framework

This script strictly implements the dual-loss mechanism described in the
MCNN paper (Xue et al., 2021), as requested.

Implementation Details:
1.  The MCNN model has two output heads: a classification head and a similarity head.
2.  The loss function is a weighted sum of a classification loss (Lp) and a
    similarity loss (Ls), following the formula L = α*Lp + β*Ls.
3.  The dataset is assumed to be class-balanced.
"""
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, BertModel
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
import warnings

warnings.filterwarnings('ignore')

# --- 1. 全局配置 ---
class Config:
    CSV_PATH = '../../spams_detection/datasets/crawler/LA/outputs/full_data_0617.csv'
    IMAGE_DIR = '../../spams_detection/datasets/crawler/LA/image/'
    TEXT_COLUMN = 'content'
    LABEL_COLUMN = 'is_recommended'
    IMAGE_IDS_COLUMN = 'photo_ids'

    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_TEXT_LEN = 128
    MAX_IMAGES = 4
    IMG_SIZE = 224
    
    BERT_DIM = 768
    RESNET_DIM = 2048
    GRU_HIDDEN_DIM = 256
    SHARED_DIM = 256

    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    
    # 论文中提到的联合损失权重
    ALPHA = 0.7  # 分类损失的权重
    BETA = 0.3   # 相似度损失的权重

    RANDOM_STATE = 42

# --- 2. 设备检查 ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Using device: {device} ---")

# --- 3. 多模态数据集类 (无变化) ---
class YelpMultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, config):
        self.df, self.tokenizer, self.image_transform, self.config = dataframe, tokenizer, image_transform, config
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.config.TEXT_COLUMN])
        text_encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.config.MAX_TEXT_LEN,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        image_ids_str = row.get(self.config.IMAGE_IDS_COLUMN)
        image_tensors = []
        if pd.notna(image_ids_str):
            image_ids = image_ids_str.split('#')[:self.config.MAX_IMAGES]
            for img_id in image_ids:
                img_path = os.path.join(self.config.IMAGE_DIR, f"{img_id}.jpg")
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensors.append(self.image_transform(image))
                except (FileNotFoundError, IOError): continue
        while len(image_tensors) < self.config.MAX_IMAGES:
            image_tensors.append(torch.zeros((3, self.config.IMG_SIZE, self.config.IMG_SIZE)))
        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'images': torch.stack(image_tensors),
            'label': torch.tensor(row[self.config.LABEL_COLUMN], dtype=torch.long)
        }

# --- 4. MCNN 模型架构 (多任务学习版) ---
class MCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- 特征提取器 (与之前相同) ---
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        self.text_gru = nn.GRU(config.BERT_DIM, config.GRU_HIDDEN_DIM, 1, batch_first=True, bidirectional=True)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.image_gru = nn.GRU(config.RESNET_DIM, config.GRU_HIDDEN_DIM, 1, batch_first=True, bidirectional=True)
        
        # 冻结预训练模型
        for p in self.bert.parameters(): p.requires_grad = False
        for p in self.resnet.parameters(): p.requires_grad = False

        # --- 相似度模块 (与之前相同) ---
        self.shared_fc = nn.Linear(config.GRU_HIDDEN_DIM * 2, config.SHARED_DIM)
        
        # --- 两个独立的输出头 ---
        # 1. 相似度头 (Similarity Head)
        self.similarity_head = nn.Sequential(
            nn.Linear(config.SHARED_DIM * 2, 1), # 输入拼接后的图文共享特征，输出一个相似度logit
            nn.Sigmoid() # 将logit压缩到0-1之间
        )

        # 2. 分类头 (Classification Head)
        fusion_input_dim = (config.GRU_HIDDEN_DIM * 2) * 2 # 仅融合图文特征
        self.classification_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, images):
        # 1. 文本特征提取
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        _, text_hidden = self.text_gru(text_features)
        text_feat_vec = torch.cat([text_hidden[0], text_hidden[1]], dim=1)

        # 2. 图像特征提取
        batch_size = images.shape[0]
        image_features_flat = self.resnet(images.view(-1, 3, self.config.IMG_SIZE, self.config.IMG_SIZE))
        image_features_seq = image_features_flat.view(batch_size, self.config.MAX_IMAGES, -1)
        _, image_hidden = self.image_gru(image_features_seq)
        image_feat_vec = torch.cat([image_hidden[0], image_hidden[1]], dim=1)
        
        # 3. 相似度预测
        text_shared = self.shared_fc(text_feat_vec)
        image_shared = self.shared_fc(image_feat_vec)
        # 论文中没有明确说明相似度头的输入，一种合理的实现是拼接共享特征
        similarity_input = torch.cat([text_shared, image_shared], dim=1)
        similarity_pred = self.similarity_head(similarity_input).squeeze(-1) # (batch_size)
        
        # 4. 分类预测
        classification_input = torch.cat([text_feat_vec, image_feat_vec], dim=1)
        classification_logits = self.classification_head(classification_input)
        
        return classification_logits, similarity_pred

# --- 5. 训练和评估流程 (已修改以支持多任务) ---
def train_epoch(model, loader, optimizer, classification_loss_fn, similarity_loss_fn, device, alpha, beta):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        classification_logits, similarity_pred = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            images=batch['images'].to(device)
        )
        
        labels = batch['label'].to(device)
        
        # 计算分类损失 (Lp)
        loss_p = classification_loss_fn(classification_logits, labels)
        
        # 计算相似度损失 (Ls)
        # 目标：真实评论(1)应该有高相似度(接近1)，虚假评论(0)应该有低相似度(接近0)
        # 所以，相似度任务的目标标签就是主任务的标签
        similarity_target = labels.float()
        loss_s = similarity_loss_fn(similarity_pred, similarity_target)
        
        # 计算联合损失
        total_batch_loss = alpha * loss_p + beta * loss_s
        
        total_batch_loss.backward()
        optimizer.step()
        total_loss += total_batch_loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # 评估时，我们只关心分类头的输出
            classification_logits, _ = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                images=batch['images'].to(device)
            )
            preds = torch.argmax(classification_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
            
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

# --- 6. 主执行流程 ---
if __name__ == '__main__':
    print("\n--- Step 1: Loading and Preparing Data ---")
    df = pd.read_csv(Config.CSV_PATH)
    # 只使用带图片的评论
    df = df[df[Config.IMAGE_IDS_COLUMN].notna()].copy()
    
    # 既然数据集是平衡的，我们可以直接使用，无需额外采样
    print(f"Using full balanced dataset of {len(df)} samples with images.")
    print("Class distribution:")
    print(df[Config.LABEL_COLUMN].value_counts(normalize=True))
    
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=Config.RANDOM_STATE, stratify=df[Config.LABEL_COLUMN])

    tokenizer = AutoTokenizer.from_pretrained(Config.PRE_TRAINED_MODEL_NAME)
    image_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = YelpMultimodalDataset(df_train, tokenizer, image_transform, Config)
    test_dataset = YelpMultimodalDataset(df_test, tokenizer, image_transform, Config)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)

    print("\n--- Step 2: Initializing MCNN Model (Multi-task Version) ---")
    model = MCNN(Config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    # 为两个任务准备不同的损失函数
    classification_loss_fn = nn.CrossEntropyLoss()
    similarity_loss_fn = nn.BCELoss() # 二元交叉熵，适用于0-1之间的相似度预测

    print("\n--- Step 3: Starting Training ---")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_epoch(
            model, train_loader, optimizer,
            classification_loss_fn, similarity_loss_fn,
            device, Config.ALPHA, Config.BETA
        )
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Avg. Joint Training Loss: {avg_loss:.4f}")

    print("\n--- Step 4: Evaluating on Test Set ---")
    accuracy, f1 = evaluate(model, test_loader, device)
    
    print("\n--- Final MCNN (Multi-task) Baseline Results ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-score (Macro): {f1:.4f}")
    print("--------------------------------------------------")