import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from collections import defaultdict


# --- 1. 配置 ---
class Config:
    # 数据路径
    DATA_PATH = '../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv'
    IMAGE_DIR = '../spams_detection/spam_datasets/crawler/LA/images/'
    ENCODER_WEIGHTS_PATH = 'conv_autoencoder_encoder.pth'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型超参数
    IMG_SIZE = 128
    EMBED_DIM = 100
    TEXT_HIDDEN_DIM = 128
    VISUAL_FEATURE_DIM = 256  # 自编码器输出的特征维度
    THRESHOLD = 0.5  # 急性阈值

    # 训练超参数
    PRETRAIN_EPOCHS = 5
    PRETRAIN_LR = 0.001
    TRAIN_EPOCHS = 30
    TRAIN_LR = 0.001
    BATCH_SIZE = 32
    ALPHA = 1.0  # 主损失权重
    BETA = 1.0  # 对抗损失权重


print(f"Using device: {Config.DEVICE}")


# --- 2. 数据集处理 ---
class ImageTextDataset(Dataset):
    def __init__(self, df, config, tokenizer, vocab):
        self.df = df
        self.config = config
        self.tokenizer = tokenizer
        self.vocab = vocab

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['content']
        label = row['label']

        # 处理图片
        image = self._load_image(row['photo_ids'])

        # 处理文本
        tokens = self.tokenizer(text.lower())
        token_ids = [self.vocab.get(t, 0) for t in tokens]  # 0 for <unk>

        return image, torch.LongTensor(token_ids), torch.tensor(label, dtype=torch.long)

    def _load_image(self, photo_ids):
        if pd.isna(photo_ids) or not isinstance(photo_ids, str) or photo_ids == '':
            return torch.zeros(3, self.config.IMG_SIZE, self.config.IMG_SIZE)

        first_photo_id = photo_ids.split('#')[0]
        img_path = os.path.join(self.config.IMAGE_DIR, f"{first_photo_id}.jpg")

        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except (FileNotFoundError, UnidentifiedImageError):
            return torch.zeros(3, self.config.IMG_SIZE, self.config.IMG_SIZE)


def custom_collate_fn(batch):
    images, texts, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)

    # 填充文本序列
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

    return images, texts_padded, labels


# --- 3. 视觉特征提取器 (VFE) ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, Config.VISUAL_FEATURE_DIM)
        )
        # Decoder
        self.decoder_fc = nn.Linear(Config.VISUAL_FEATURE_DIM, 128 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Output range [-1, 1] to match normalization
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded_latent = F.relu(self.decoder_fc(encoded))
        decoded_latent = decoded_latent.view(-1, 128, 16, 16)
        decoded = self.decoder(decoded_latent)
        return encoded, decoded


# --- 4. 文本特征提取器 (TFE) ---
class TFE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)

        attn_weights = F.softmax(self.attention(gru_out), dim=1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)
        return context_vector


# --- 5. 核心模型: mEXACT ---
class AcuteThresholding(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        # x is a feature matrix (batch_size, feature_dim)
        significance_scores = F.softmax(x, dim=-1)

        # Create mask for Least Significant (LS) features
        mask_ls = (significance_scores < self.threshold).float()

        features_ls = x * mask_ls
        return features_ls


class mEXACT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config

        # VFE
        autoencoder = ConvAutoencoder()
        # 加载预训练的编码器权重
        autoencoder.encoder.load_state_dict(torch.load(config.ENCODER_WEIGHTS_PATH))
        self.vfe = autoencoder.encoder
        # 冻结VFE的参数
        for param in self.vfe.parameters():
            param.requires_grad = False

        # TFE
        self.tfe = TFE(vocab_size, config.EMBED_DIM, config.TEXT_HIDDEN_DIM)

        # Acute Thresholding
        self.at_visual = AcuteThresholding(config.THRESHOLD)
        self.at_textual = AcuteThresholding(config.THRESHOLD)

        # Prediction
        combined_dim = config.VISUAL_FEATURE_DIM + config.TEXT_HIDDEN_DIM * 2
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(combined_dim // 2, 1)
        )

    def forward(self, images, texts):
        # 1. Feature Extraction
        f_v = self.vfe(images)
        f_t = self.tfe(texts)

        # 2. Acute Thresholding
        f_ls_v = self.at_visual(f_v)
        f_ls_t = self.at_textual(f_t)

        # 3. Model Prediction
        # First output terminal (full features)
        combined_full = torch.cat([f_v, f_t], dim=1)
        y_hat = self.fc(combined_full)

        # Second output terminal (LS features)
        combined_ls = torch.cat([f_ls_v, f_ls_t], dim=1)
        y_hat_prime = self.fc(combined_ls)

        return torch.sigmoid(y_hat), torch.sigmoid(y_hat_prime)


# --- 6. 训练与评估 ---
def pretrain_autoencoder(config):
    print("--- Phase 1: Pre-training Convolutional Autoencoder ---")
    df = pd.read_csv(config.DATA_PATH)
    # 仅使用有有效图片ID的行进行预训练
    df.dropna(subset=['photo_ids'], inplace=True)

    # 创建一个仅用于图像的数据集
    class ImageOnlyDataset(Dataset):
        def __init__(self, df, config):
            self.df = df
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            if pd.isna(row['photo_ids']) or not isinstance(row['photo_ids'], str) or row['photo_ids'] == '':
                return torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE)

            first_photo_id = row['photo_ids'].split('#')[0]
            img_path = os.path.join(config.IMAGE_DIR, f"{first_photo_id}.jpg")
            try:
                image = Image.open(img_path).convert('RGB')
                return self.transform(image)
            except (FileNotFoundError, UnidentifiedImageError):
                return torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE)

    dataset = ImageOnlyDataset(df, config)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = ConvAutoencoder().to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.PRETRAIN_LR)

    for epoch in range(config.PRETRAIN_EPOCHS):
        total_loss = 0
        for images in tqdm(loader, desc=f"Pre-train Epoch {epoch + 1}/{config.PRETRAIN_EPOCHS}"):
            images = images.to(config.DEVICE)
            _, decoded = model(images)
            loss = criterion(decoded, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Reconstruction Loss: {total_loss / len(loader):.4f}")

    torch.save(model.encoder.state_dict(), config.ENCODER_WEIGHTS_PATH)
    print(f"Saved pre-trained encoder weights to {config.ENCODER_WEIGHTS_PATH}")


def main():
    config = Config()

    # --- 阶段一: 预训练VFE ---
    if not os.path.exists(config.ENCODER_WEIGHTS_PATH):
        pretrain_autoencoder(config)
    else:
        print("Found pre-trained encoder weights. Skipping pre-training.")

    # --- 阶段二: 训练主分类模型 ---
    print("\n--- Phase 2: Training mEXACT Classifier ---")
    df = pd.read_csv(config.DATA_PATH)
    df = df[df['is_recommended'].isin([0, 1])].copy()
    df['label'] = 1 - df['is_recommended'].astype(int)
    df['content'] = df['content'].fillna('').astype(str)

    # 构建词汇表
    all_tokens = [t for text in df['content'] for t in word_tokenize(text.lower())]
    vocab = defaultdict(lambda: 0)  # 0 is <unk>
    for i, word in enumerate(sorted(list(set(all_tokens)))):
        vocab[word] = i + 1  # 1 onwards
    vocab_size = len(vocab) + 1

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.125, stratify=train_df['label'], random_state=42)

    train_dataset = ImageTextDataset(train_df, config, word_tokenize, vocab)
    val_dataset = ImageTextDataset(val_df, config, word_tokenize, vocab)
    test_dataset = ImageTextDataset(test_df, config, word_tokenize, vocab)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, collate_fn=custom_collate_fn)

    model = mEXACT(vocab_size, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_LR)
    criterion = nn.BCELoss()  # Binary Cross Entropy

    for epoch in range(config.TRAIN_EPOCHS):
        model.train()
        total_loss = 0
        for images, texts, labels in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{config.TRAIN_EPOCHS}"):
            images, texts, labels = images.to(config.DEVICE), texts.to(config.DEVICE), labels.to(
                config.DEVICE).float().unsqueeze(1)

            y_hat, y_hat_prime = model(images, texts)

            loss_main = criterion(y_hat, labels)
            loss_adv = criterion(y_hat_prime, 1 - labels)  # 对抗损失

            loss = config.ALPHA * loss_main + config.BETA * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, texts, labels in val_loader:
                images, texts = images.to(config.DEVICE), texts.to(config.DEVICE)
                y_hat, _ = model(images, texts)
                preds = (y_hat.squeeze() > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        print(
            f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

    # 最终测试
    print("\n--- Final Evaluation on Test Set ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, texts, labels in tqdm(test_loader, desc="Testing"):
            images, texts = images.to(config.DEVICE), texts.to(config.DEVICE)
            y_hat, _ = model(images, texts)
            preds = (y_hat.squeeze() > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")


if __name__ == '__main__':
    main()