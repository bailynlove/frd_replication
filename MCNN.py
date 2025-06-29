# -*- coding: utf-8 -*-
"""
Replication of MCNN for Fake Review Detection on Yelp Dataset (Corrected Version)

This script implements the core components of the MCNN paper (Xue et al., 2021)
and adapts them for the Yelp multimodal review dataset.

Correction Log:
-   Fixed the critical bug in MCNN.__init__ where a Tokenizer was used instead of a Model.
-   The MCNN class now correctly loads the BertModel and freezes its parameters.
-   The Tokenizer is now correctly initialized in the main script flow for the Dataset.
"""
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, BertModel  # <-- 显式导入 BertModel
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
    # 路径 (根据您的设置)
    CSV_PATH = '../../spams_detection/datasets/crawler/LA/outputs/full_data_0617.csv'
    IMAGE_DIR = '../../spams_detection/datasets/crawler/LA/image/'
    TEXT_COLUMN = 'content'
    LABEL_COLUMN = 'is_recommended'
    IMAGE_IDS_COLUMN = 'photo_ids'

    # 模型超参数
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_TEXT_LEN = 128
    MAX_IMAGES = 4
    IMG_SIZE = 224
    
    BERT_DIM = 768
    RESNET_DIM = 2048
    GRU_HIDDEN_DIM = 256
    SHARED_DIM = 256

    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 1e-4

    # 数据集划分
    TRAIN_SIZE = 0.7
    RANDOM_STATE = 42

# --- 2. 设备检查 (根据您的设置) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ CUDA is available! Using CUDA as the device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ MPS is available! Using MPS as the device.")
else:
    device = torch.device("cpu")
    print("⚠️ CUDA/MPS not available, using CPU.")


# --- 3. 多模态数据集类 (无变化) ---
class YelpMultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, config):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 文本处理
        text = str(row[self.config.TEXT_COLUMN])
        text_encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.config.MAX_TEXT_LEN,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        # 图像处理
        image_ids_str = row.get(self.config.IMAGE_IDS_COLUMN)
        image_tensors = []
        if pd.notna(image_ids_str):
            image_ids = image_ids_str.split('#')
            for img_id in image_ids[:self.config.MAX_IMAGES]:
                img_path = os.path.join(self.config.IMAGE_DIR, f"{img_id}.jpg")
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensors.append(self.image_transform(image))
                except (FileNotFoundError, IOError):
                    continue
        
        num_images_found = len(image_tensors)
        while len(image_tensors) < self.config.MAX_IMAGES:
            image_tensors.append(torch.zeros((3, self.config.IMG_SIZE, self.config.IMG_SIZE)))

        images = torch.stack(image_tensors)
        
        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'images': images,
            'label': torch.tensor(row[self.config.LABEL_COLUMN], dtype=torch.long)
        }

# --- 4. MCNN 模型架构 (已修正) ---
class MCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- 文本子网络 ---
        # 【关键修正点】加载 BertModel 而不是 Tokenizer
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        # 现在可以正确地冻结模型参数了
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.text_gru = nn.GRU(
            input_size=config.BERT_DIM, hidden_size=config.GRU_HIDDEN_DIM,
            num_layers=1, batch_first=True, bidirectional=True
        )
        
        # --- 图像子网络 ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # 使用推荐的权重加载方式
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        self.image_gru = nn.GRU(
            input_size=config.RESNET_DIM, hidden_size=config.GRU_HIDDEN_DIM,
            num_layers=1, batch_first=True, bidirectional=True
        )

        # --- 相似度模块 ---
        self.shared_fc = nn.Linear(config.GRU_HIDDEN_DIM * 2, config.SHARED_DIM)
        
        # --- 融合与分类模块 ---
        fusion_input_dim = (config.GRU_HIDDEN_DIM * 2) * 2 + 1
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, images):
        # 1. 文本特征提取
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        _, text_hidden = self.text_gru(text_features)
        text_feat_vec = torch.cat([text_hidden[0], text_hidden[1]], dim=1)

        # 2. 图像特征提取
        batch_size = images.shape[0]
        image_features_flat = self.resnet(images.view(-1, 3, self.config.IMG_SIZE, self.config.IMG_SIZE))
        image_features_flat = image_features_flat.view(image_features_flat.size(0), -1)
        image_features_seq = image_features_flat.view(batch_size, self.config.MAX_IMAGES, -1)
        _, image_hidden = self.image_gru(image_features_seq)
        image_feat_vec = torch.cat([image_hidden[0], image_hidden[1]], dim=1)
        
        # 3. 相似度计算
        text_shared = self.shared_fc(text_feat_vec)
        image_shared = self.shared_fc(image_feat_vec)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(text_shared, image_shared).unsqueeze(1)
        
        # 4. 特征融合与分类
        fused_features = torch.cat([text_feat_vec, image_feat_vec, similarity], dim=1)
        logits = self.fusion_classifier(fused_features)
        
        return logits

# --- 5. 训练和评估流程 ---
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        logits = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            images=batch['images'].to(device)
        )
        loss = loss_fn(logits, batch['label'].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                images=batch['images'].to(device)
            )
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

# --- 6. 主执行流程 ---
if __name__ == '__main__':
    print("\n--- Step 1: Loading and Preparing Data ---")
    try:
        df = pd.read_csv(Config.CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {Config.CSV_PATH}")
        exit()
        
    df = df[df[Config.IMAGE_IDS_COLUMN].notna()].copy()
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=Config.RANDOM_STATE)
    df = df.reset_index(drop=True)
    print(f"Using {len(df)} samples with images for demonstration.")
    
    df_train, df_test = train_test_split(df, train_size=Config.TRAIN_SIZE, random_state=Config.RANDOM_STATE, stratify=df[Config.LABEL_COLUMN])
    
    # 【关键修正点】Tokenizer 在这里初始化
    tokenizer = AutoTokenizer.from_pretrained(Config.PRE_TRAINED_MODEL_NAME)
    image_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = YelpMultimodalDataset(df_train, tokenizer, image_transform, Config)
    test_dataset = YelpMultimodalDataset(df_test, tokenizer, image_transform, Config)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)

    print("\n--- Step 2: Initializing MCNN Model ---")
    model = MCNN(Config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print("\n--- Step 3: Starting Training ---")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Avg. Training Loss: {avg_loss:.4f}")

    print("\n--- Step 4: Evaluating on Test Set ---")
    accuracy, f1 = evaluate(model, test_loader, device)
    
    print("\n--- Final MCNN Baseline Results ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-score (Macro): {f1:.4f}")
    print("-----------------------------------")