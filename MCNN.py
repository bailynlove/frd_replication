# -*- coding: utf-8 -*-
"""
Strict Replication of MCNN with Multi-task Learning Framework (Enhanced Version)

This script strictly implements the dual-loss mechanism from the MCNN paper.

Enhancements:
1.  **Comprehensive Metrics**: Evaluation now includes Accuracy, Macro F1-score,
    and Precision/Recall/F1-score specifically for the 'Fake' class.
2.  **Best Model Saving**: During training, the model with the best F1-score
    on the validation set is saved.
3.  **Load & Test**: The script will load the best saved model for final testing.
4.  **Result Persistence**: Test predictions and labels are saved to a CSV file
    for future analysis without re-training.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import os
import warnings
import json

warnings.filterwarnings('ignore')

# --- 1. 全局配置 ---
class Config:
    CSV_PATH = '../../spams_detection/datasets/crawler/LA/outputs/full_data_0617.csv'
    IMAGE_DIR = '../../spams_detection/datasets/crawler/LA/image/'
    TEXT_COLUMN = 'content'
    LABEL_COLUMN = 'is_recommended'
    IMAGE_IDS_COLUMN = 'photo_ids'

    # 定义哪个标签代表“虚假评论”，以便计算相关指标
    FAKE_CLASS_LABEL = 0 

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
    
    ALPHA = 0.7
    BETA = 0.3

    RANDOM_STATE = 42
    
    # 文件保存路径
    MODEL_SAVE_PATH = "./best_mcnn_model.pth"
    RESULTS_SAVE_PATH = "./mcnn_test_predictions.csv"

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
        self.config = config
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        self.text_gru = nn.GRU(config.BERT_DIM, config.GRU_HIDDEN_DIM, 1, batch_first=True, bidirectional=True)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.image_gru = nn.GRU(config.RESNET_DIM, config.GRU_HIDDEN_DIM, 1, batch_first=True, bidirectional=True)
        for p in self.bert.parameters(): p.requires_grad = False
        for p in self.resnet.parameters(): p.requires_grad = False
        self.shared_fc = nn.Linear(config.GRU_HIDDEN_DIM * 2, config.SHARED_DIM)
        self.similarity_head = nn.Sequential(
            nn.Linear(config.SHARED_DIM * 2, 1), nn.Sigmoid()
        )
        fusion_input_dim = (config.GRU_HIDDEN_DIM * 2) * 2
        self.classification_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 512), nn.ReLU(),
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


# --- 5. 训练和评估流程 (已增强) ---
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
        loss_p = classification_loss_fn(classification_logits, labels)
        similarity_target = labels.float()
        loss_s = similarity_loss_fn(similarity_pred, similarity_target)
        total_batch_loss = alpha * loss_p + beta * loss_s
        total_batch_loss.backward()
        optimizer.step()
        total_loss += total_batch_loss.item()
    return total_loss / len(loader)

def get_predictions(model, loader, device):
    """只获取预测结果和真实标签，用于评估。"""
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
    return np.array(all_labels), np.array(all_preds)

def print_metrics(y_true, y_pred, fake_class_label):
    """打印所有需要的指标。"""
    print("\n--- Overall Performance ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.4f}")
    
    print(f"\n--- Metrics for FAKE Class (Label: {fake_class_label}) ---")
    precision = precision_score(y_true, y_pred, pos_label=fake_class_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=fake_class_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=fake_class_label, zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print("\n--- Full Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Fake (0)', 'Real (1)'], zero_division=0))


# --- 6. 主执行流程 ---
if __name__ == '__main__':
    # --- 数据加载与准备 ---
    print("\n--- Step 1: Loading and Preparing Data ---")
    df = pd.read_csv(Config.CSV_PATH)
    df = df[df[Config.IMAGE_IDS_COLUMN].notna()].copy().reset_index(drop=True)
    print(f"Using {len(df)} samples with images.")
    print("Class distribution:\n", df[Config.LABEL_COLUMN].value_counts(normalize=True))
    
    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=Config.RANDOM_STATE, stratify=df[Config.LABEL_COLUMN])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=Config.RANDOM_STATE, stratify=df_temp[Config.LABEL_COLUMN])

    tokenizer = AutoTokenizer.from_pretrained(Config.PRE_TRAINED_MODEL_NAME)
    image_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = YelpMultimodalDataset(df_train, tokenizer, image_transform, Config)
    val_dataset = YelpMultimodalDataset(df_val, tokenizer, image_transform, Config)
    test_dataset = YelpMultimodalDataset(df_test, tokenizer, image_transform, Config)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)

    # --- 模型训练与保存 ---
    print("\n--- Step 2: Training MCNN Model ---")
    model = MCNN(Config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    classification_loss_fn = nn.CrossEntropyLoss()
    similarity_loss_fn = nn.BCELoss()

    best_val_f1 = -1.0

    for epoch in range(Config.EPOCHS):
        avg_loss = train_epoch(
            model, train_loader, optimizer,
            classification_loss_fn, similarity_loss_fn,
            device, Config.ALPHA, Config.BETA
        )
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Avg. Joint Training Loss: {avg_loss:.4f}")
        
        # 在验证集上评估并保存最佳模型
        val_labels, val_preds = get_predictions(model, val_loader, device)
        val_f1_fake_class = f1_score(val_labels, val_preds, pos_label=Config.FAKE_CLASS_LABEL, average='binary')
        
        print(f"Validation F1-score (Fake Class): {val_f1_fake_class:.4f}")
        
        if val_f1_fake_class > best_val_f1:
            best_val_f1 = val_f1_fake_class
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✨ New best model saved to {Config.MODEL_SAVE_PATH} with F1-score: {best_val_f1:.4f}")

    # --- 模型评估与结果保存 ---
    print("\n--- Step 3: Evaluating Best Model on Test Set ---")
    # 加载表现最好的模型
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    print(f"Loaded best model from {Config.MODEL_SAVE_PATH}")
    
    # 获取测试集预测结果
    test_labels, test_preds = get_predictions(model, test_loader, device)
    
    # 打印所有指标
    print_metrics(test_labels, test_preds, Config.FAKE_CLASS_LABEL)
    
    # 保存预测结果和真实标签到CSV文件
    results_df = pd.DataFrame({
        'true_label': test_labels,
        'predicted_label': test_preds
    })
    # 可以将原始文本也加入，方便分析
    results_df = pd.concat([df_test.reset_index(drop=True), results_df], axis=1)
    results_df.to_csv(Config.RESULTS_SAVE_PATH, index=False)
    print(f"\n✅ Test predictions and labels saved to {Config.RESULTS_SAVE_PATH}")