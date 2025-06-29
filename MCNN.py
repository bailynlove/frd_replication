# -*- coding: utf-8 -*-
"""
Strict Replication of MCNN with Data Balancing (Final Version for Consistent Metrics)

This script implements the MCNN model with oversampling to handle class imbalance.

Final Evaluation Logic:
-   Metrics are calculated using the standard 'binary' average setting in scikit-learn.
-   This means Precision, Recall, and F1-score are calculated for the positive class (label=1, 'Real Class').
-   This ensures consistency for comparison with other models.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
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
    
    # 在二元计算中，1被认为是正类 (positive class)
    POSITIVE_CLASS_LABEL = 1 

    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_TEXT_LEN, MAX_IMAGES, IMG_SIZE = 128, 4, 224
    BERT_DIM, RESNET_DIM, GRU_HIDDEN_DIM, SHARED_DIM = 768, 2048, 256, 256
    BATCH_SIZE, EPOCHS, LEARNING_RATE = 8, 5, 1e-4
    ALPHA, BETA = 0.7, 0.3
    RANDOM_STATE = 42
    MODEL_SAVE_PATH = "./best_mcnn_model_balanced.pth"

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

# --- 4. MCNN 模型架构 (无变化) ---
class MCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.bert = config, BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        self.text_gru = nn.GRU(config.BERT_DIM, config.GRU_HIDDEN_DIM, 1, batch_first=True, bidirectional=True)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.image_gru = nn.GRU(config.RESNET_DIM, config.GRU_HIDDEN_DIM, 1, batch_first=True, bidirectional=True)
        for p in self.bert.parameters(): p.requires_grad = False
        for p in self.resnet.parameters(): p.requires_grad = False
        self.shared_fc = nn.Linear(config.GRU_HIDDEN_DIM * 2, config.SHARED_DIM)
        self.similarity_head = nn.Sequential(nn.Linear(config.SHARED_DIM * 2, 1), nn.Sigmoid())
        self.classification_head = nn.Sequential(
            nn.Linear((config.GRU_HIDDEN_DIM * 2) * 2, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, 2)
        )
    def forward(self, input_ids, attention_mask, images):
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        _, text_hidden = self.text_gru(text_features)
        text_feat_vec = torch.cat([text_hidden[0], text_hidden[1]], dim=1)
        batch_size = images.shape[0]
        image_features_flat = self.resnet(images.view(-1, 3, self.config.IMG_SIZE, self.config.IMG_SIZE))
        image_features_seq = image_features_flat.view(batch_size, self.config.MAX_IMAGES, -1)
        _, image_hidden = self.image_gru(image_features_seq)
        image_feat_vec = torch.cat([image_hidden[0], image_hidden[1]], dim=1)
        text_shared, image_shared = self.shared_fc(text_feat_vec), self.shared_fc(image_feat_vec)
        similarity_pred = self.similarity_head(torch.cat([text_shared, image_shared], dim=1)).squeeze(-1)
        classification_logits = self.classification_head(torch.cat([text_feat_vec, image_feat_vec], dim=1))
        return classification_logits, similarity_pred


# --- 5. 训练和评估流程 (评估函数已修改) ---
def train_epoch(model, loader, optimizer, classification_loss_fn, similarity_loss_fn, device, alpha, beta):
    model.train()
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        classification_logits, similarity_pred = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            images=batch['images'].to(device)
        )
        labels = batch['label'].to(device)
        loss_p = classification_loss_fn(classification_logits, labels)
        loss_s = similarity_loss_fn(similarity_pred, labels.float())
        total_batch_loss = alpha * loss_p + beta * loss_s
        total_batch_loss.backward()
        optimizer.step()

def get_predictions_and_evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            classification_logits, _ = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                images=batch['images'].to(device)
            )
            preds = torch.argmax(classification_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    
    # --- 【关键修改点】计算您要求的常规指标 ---
    # `average='binary'` 是默认行为，计算的是正类（label=1）的指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # 我们也计算宏平均F1作为参考
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return accuracy, precision, recall, f1, macro_f1


# --- 6. 主执行流程 ---
if __name__ == '__main__':
    # Step 1: 数据加载与过滤
    print("\n--- Step 1: Loading and Filtering Data ---")
    df = pd.read_csv(Config.CSV_PATH)
    df = df[df[Config.IMAGE_IDS_COLUMN].notna()].copy().reset_index(drop=True)
    print(f"Filtered to {len(df)} samples with images.")
    print("Initial class distribution:\n", df[Config.LABEL_COLUMN].value_counts(normalize=True))
    
    # Step 2: 数据集划分
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=Config.RANDOM_STATE, stratify=df[Config.LABEL_COLUMN])
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=Config.RANDOM_STATE, stratify=df_test[Config.LABEL_COLUMN])
    
    # Step 3: 对训练集进行过采样
    print("\n--- Step 3: Applying RandomOversampling to the Training Set ---")
    ros = RandomOverSampler(random_state=Config.RANDOM_STATE)
    train_indices = df_train.index.to_numpy().reshape(-1, 1)
    train_labels = df_train[Config.LABEL_COLUMN].to_numpy()
    resampled_indices, _ = ros.fit_resample(train_indices, train_labels)
    df_train_balanced = df.iloc[resampled_indices.flatten()].reset_index(drop=True)
    print("Balanced training set distribution:\n", df_train_balanced[Config.LABEL_COLUMN].value_counts())
    
    # Step 4: 创建DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(Config.PRE_TRAINED_MODEL_NAME)
    image_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = YelpMultimodalDataset(df_train_balanced, tokenizer, image_transform, Config)
    val_dataset = YelpMultimodalDataset(df_val, tokenizer, image_transform, Config)
    test_dataset = YelpMultimodalDataset(df_test, tokenizer, image_transform, Config)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)

    # Step 5: 模型训练与保存
    print("\n--- Step 5: Training MCNN Model ---")
    model = MCNN(Config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    classification_loss_fn = nn.CrossEntropyLoss()
    similarity_loss_fn = nn.BCELoss()

    best_val_f1 = -1.0
    for epoch in range(Config.EPOCHS):
        train_epoch(model, train_loader, optimizer, classification_loss_fn, similarity_loss_fn, device, Config.ALPHA, Config.BETA)
        
        # 我们使用宏平均F1来选择最佳模型，因为它能均衡地反映模型对两个类的识别能力
        _, _, _, _, val_macro_f1 = get_predictions_and_evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Validation Macro F1-score: {val_macro_f1:.4f}")
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✨ New best model saved with Macro F1: {best_val_f1:.4f}")

    # Step 6: 最终评估
    print("\n--- Step 6: Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    
    accuracy, precision, recall, f1, _ = get_predictions_and_evaluate(model, test_loader, device)
    
    print("\n--- FINAL MCNN PERFORMANCE (Standard Binary Metrics) ---")
    print("Metrics are calculated for the positive class (label=1)")
    print("="*55)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("="*55)