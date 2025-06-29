# -*- coding: utf-8 -*-
"""
Replication of MPFN (Multimodal Progressive Fusion Network)

This script implements the core idea of the MPFN paper (Jing et al., 2023),
which is the progressive fusion of features at different hierarchical levels.

Key Implementation Details:
1.  **Hierarchical Feature Extraction**:
    -   Text: BERT is used to extract a single, powerful deep text feature vector.
    -   Image: A pre-trained CNN (EfficientNet-B0) is used to extract features
        from its shallow, intermediate, and deep layers to simulate the
        multi-level visual features described in the paper.
2.  **Progressive Fusion**:
    -   Multiple "Mixer" modules are created to fuse the text feature with
        visual features from different depths.
3.  **Final Aggregation & Classification**:
    -   Features from all fusion stages are concatenated and passed to a final
        classifier to make the prediction.
4.  **Data Balancing**: The script continues to use RandomOversampling on the
    training set to handle the severe class imbalance.
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
    POSITIVE_CLASS_LABEL = 1

    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_TEXT_LEN, MAX_IMAGES, IMG_SIZE = 128, 1, 224 # MPFN通常处理单图，我们简化为只用第一张图
    
    # 特征维度
    BERT_DIM = 768
    # 我们将从EfficientNet的不同层提取特征
    IMG_SHALLOW_DIM = 24  # from layer 2
    IMG_MID_DIM = 112     # from layer 5
    IMG_DEEP_DIM = 1280   # from final layer
    
    FUSION_DIM = 256 # 融合后的维度

    BATCH_SIZE, EPOCHS, LEARNING_RATE = 16, 5, 5e-5
    RANDOM_STATE = 42
    MODEL_SAVE_PATH = "./best_mpfn_model.pth"

# --- 2. 设备检查 ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Using device: {device} ---")


# --- 3. 数据集类 (已修改为只加载第一张图) ---
class YelpProgressiveDataset(Dataset):
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
        
        image_tensor = torch.zeros((3, self.config.IMG_SIZE, self.config.IMG_SIZE))
        image_ids_str = row.get(self.config.IMAGE_IDS_COLUMN)
        if pd.notna(image_ids_str):
            first_img_id = image_ids_str.split('#')[0]
            img_path = os.path.join(self.config.IMAGE_DIR, f"{first_img_id}.jpg")
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.image_transform(image)
            except (FileNotFoundError, IOError): pass
            
        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'image': image_tensor,
            'label': torch.tensor(row[self.config.LABEL_COLUMN], dtype=torch.long)
        }

# --- 4. MPFN 模型架构 ---
class MlpMixer(nn.Module):
    """一个简化的MLP Mixer，用于融合图文特征"""
    def __init__(self, text_dim, image_dim, output_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.mixer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    def forward(self, text_feat, image_feat):
        text_proj = self.text_proj(text_feat)
        image_proj = self.image_proj(image_feat)
        combined = torch.cat((text_proj, image_proj), dim=1)
        return self.mixer(combined)

class MPFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 文本提取器
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        
        # 图像提取器 (使用EfficientNet-B0，因为它分层清晰)
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # 冻结所有权重，只用作特征提取
        for param in effnet.parameters():
            param.requires_grad = False
        
        self.effnet_layers = effnet.features
        self.final_pool = effnet.avgpool
        
        # 渐进式融合模块
        self.mixer_shallow = MlpMixer(config.BERT_DIM, config.IMG_SHALLOW_DIM, config.FUSION_DIM)
        self.mixer_mid = MlpMixer(config.BERT_DIM, config.IMG_MID_DIM, config.FUSION_DIM)
        self.mixer_deep = MlpMixer(config.BERT_DIM, config.IMG_DEEP_DIM, config.FUSION_DIM)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, image):
        # 1. 文本特征提取
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS] token的表示作为整个文本的特征
        text_feat = text_output.pooler_output
        
        # 2. 图像分层特征提取
        # 我们需要手动通过EfficientNet的层来获取中间特征
        img_feat_s2 = self.effnet_layers[2](self.effnet_layers[1](self.effnet_layers[0](image)))
        img_feat_s5 = self.effnet_layers[5](self.effnet_layers[4](self.effnet_layers[3](img_feat_s2)))
        img_feat_s8 = self.effnet_layers[8](self.effnet_layers[7](self.effnet_layers[6](img_feat_s5)))
        
        # 对每层特征进行全局平均池化，得到特征向量
        img_shallow = nn.functional.adaptive_avg_pool2d(img_feat_s2, (1, 1)).flatten(1)
        img_mid = nn.functional.adaptive_avg_pool2d(img_feat_s5, (1, 1)).flatten(1)
        img_deep = self.final_pool(img_feat_s8).flatten(1)

        # 3. 渐进式融合
        fused_shallow = self.mixer_shallow(text_feat, img_shallow)
        fused_mid = self.mixer_mid(text_feat, img_mid)
        fused_deep = self.mixer_deep(text_feat, img_deep)
        
        # 4. 最终特征聚合与分类
        final_fused_feat = torch.cat([fused_shallow, fused_mid, fused_deep], dim=1)
        logits = self.classifier(final_fused_feat)
        
        return logits

# --- 5. 训练和评估流程 ---
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        logits = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            image=batch['image'].to(device)
        )
        loss = loss_fn(logits, batch['label'].to(device))
        loss.backward()
        optimizer.step()

def get_predictions_and_evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                image=batch['image'].to(device)
            )
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return accuracy, precision, recall, f1

# --- 6. 主执行流程 ---
if __name__ == '__main__':
    # Step 1: 数据加载与过滤
    print("\n--- Step 1: Loading and Filtering Data ---")
    df = pd.read_csv(Config.CSV_PATH)
    df = df[df[Config.IMAGE_IDS_COLUMN].notna()].copy().reset_index(drop=True)
    print(f"Filtered to {len(df)} samples with images.")
    
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

    train_dataset = YelpProgressiveDataset(df_train_balanced, tokenizer, image_transform, Config)
    val_dataset = YelpProgressiveDataset(df_val, tokenizer, image_transform, Config)
    test_dataset = YelpProgressiveDataset(df_test, tokenizer, image_transform, Config)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)

    # Step 5: 模型训练与保存
    print("\n--- Step 5: Training MPFN Model ---")
    model = MPFN(Config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    # 仍然需要处理不平衡问题，我们使用加权损失函数
    class_counts = df_train_balanced[Config.LABEL_COLUMN].value_counts().sort_index()
    weights = (len(df_train_balanced) / (2 * class_counts)).values
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using weighted loss with weights: {class_weights.cpu().numpy()}")

    best_val_f1 = -1.0
    for epoch in range(Config.EPOCHS):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        _, _, _, val_f1 = get_predictions_and_evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Validation F1-score: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✨ New best model saved with F1-score: {best_val_f1:.4f}")

    # Step 6: 最终评估
    print("\n--- Step 6: Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    
    accuracy, precision, recall, f1 = get_predictions_and_evaluate(model, test_loader, device)
    
    print("\n--- FINAL MPFN PERFORMANCE ---")
    print("Metrics are calculated for the positive class (label=1)")
    print("="*40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("="*40)