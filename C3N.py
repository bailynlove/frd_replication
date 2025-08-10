import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, BertTokenizer  # 使用CLIP的tokenizer
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torch.optim import AdamW
from PIL import Image
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import spacy

warnings.filterwarnings("ignore")

# --- 命令行参数 ---
parser = argparse.ArgumentParser(description='Train C3N Model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
args = parser.parse_args()

# --- 配置 ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "image_dir": "../spams_detection/spam_datasets/crawler/LA/images/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 5,
    "learning_rate": 1e-5,
    "clip_model": "openai/clip-vit-base-patch32",
    "num_nouns": 20,  # 提取名词的最大数量
    "num_crops": 5,  # 提取图像区域的最大数量
    "hidden_dim": 128,
    "num_classes": 2,
    "best_model_path": "c3n_best_model.pth"
}

print(f"使用设备: {CONFIG['device']}")


# --- 主模型 C3N ---
class C3N(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 特征提取器
        self.clip = CLIPModel.from_pretrained(config["clip_model"])

        # 2. 跨模态增强模块
        self.enhancement_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.clip.config.projection_dim, nhead=8, batch_first=True),
            num_layers=2
        )

        # 3. 跨模态关联模块 (1D-CNN)
        # 输入通道数 = 图像区域数+1, 输出通道数 = 64
        self.correlation_cnn = nn.Conv1d(
            in_channels=config["num_crops"] + 1,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # 4. 跨模态融合模块 (MLPs)
        fusion_input_dim = self.clip.config.projection_dim * 2 + 64  # 全局文本 + 全局图像 + 关联特征
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 5. 分类器
        self.classifier = nn.Linear(config["hidden_dim"], config["num_classes"])

    def forward(self, text_tokens, image_pixels):
        # text_tokens: (B, N, SeqLen), image_pixels: (B, M, C, H, W)
        # N = num_nouns + 1, M = num_crops + 1
        B, N, SeqLen = text_tokens.shape
        _, M, C, H, W = image_pixels.shape

        # 1. 词语和图像区域特征提取 (使用CLIP)
        # 将batch和序列维度合并，一次性送入CLIP
        text_features = self.clip.get_text_features(
            input_ids=text_tokens.view(-1, SeqLen)
        ).view(B, N, -1)  # (B, N, ProjDim)

        image_features = self.clip.get_image_features(
            pixel_values=image_pixels.view(-1, C, H, W)
        ).view(B, M, -1)  # (B, M, ProjDim)

        # 2. 跨模态增强
        enhanced_text = self.enhancement_transformer(text_features)
        enhanced_image = self.enhancement_transformer(image_features)

        # 3. 跨模态关联
        # 计算相似度矩阵
        sim_matrix = F.cosine_similarity(enhanced_text.unsqueeze(2), enhanced_image.unsqueeze(1), dim=-1)  # (B, N, M)

        # 使用1D-CNN提取关联特征
        # (B, N, M) -> (B, M, N) 以便在文本维度上卷积
        correlation_features = self.correlation_cnn(sim_matrix.permute(0, 2, 1))  # (B, 64, N)
        # 全局平均池化
        correlation_features = correlation_features.mean(dim=2)  # (B, 64)

        # 4. 跨模态融合
        global_text_feat = enhanced_text[:, 0, :]  # 全文特征
        global_image_feat = enhanced_image[:, 0, :]  # 全图特征

        fused_features = self.fusion_mlp(
            torch.cat([global_text_feat, global_image_feat, correlation_features], dim=1)
        )

        # 5. 分类
        logits = self.classifier(fused_features)
        return logits


# --- 数据集 ---
class SpamCorrelationDataset(Dataset):
    def __init__(self, df, clip_processor, nlp, object_detector):
        self.df = df
        self.clip_processor = clip_processor
        self.nlp = nlp
        self.object_detector = object_detector.to(CONFIG['device'])
        self.object_detector.eval()

    def __len__(self):
        return len(self.df)

    def _extract_nouns(self, text):
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ == 'NOUN'][:CONFIG['num_nouns']]

    def _extract_crops(self, image):
        # 转换为tensor
        from torchvision.transforms.functional import to_tensor
        img_tensor = to_tensor(image).to(CONFIG['device'])

        with torch.no_grad():
            predictions = self.object_detector([img_tensor])[0]

        # 按置信度排序并选择top-k
        top_indices = torch.topk(predictions['scores'], k=min(CONFIG['num_crops'], len(predictions['scores']))).indices
        boxes = predictions['boxes'][top_indices].cpu().numpy().astype(int)

        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            crops.append(image.crop((x1, y1, x2, y2)))
        return crops

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['content'])
        label = int(row['is_recommended'])

        image_path = ""
        photo_ids = str(row.get('photo_ids', ''))
        if photo_ids and photo_ids != 'nan':
            image_path = os.path.join(CONFIG["image_dir"], f"{photo_ids.split('#')[0]}.jpg")
        try:
            image = Image.open(image_path).convert('RGB') if os.path.exists(image_path) else Image.new('RGB',
                                                                                                       (224, 224))
        except:
            image = Image.new('RGB', (224, 224))

        # 提取名词和图像区域
        nouns = self._extract_nouns(text)
        crops = self._extract_crops(image)

        # 准备文本输入 (全文 + 名词)
        text_list = [text] + nouns
        text_inputs = self.clip_processor(text=text_list, return_tensors='pt', padding='max_length', truncation=True,
                                          max_length=77)

        # 准备图像输入 (全图 + 区域)
        image_list = [image] + crops
        image_inputs = self.clip_processor(images=image_list, return_tensors='pt', padding=True)

        return {
            'text_tokens': text_inputs['input_ids'],
            'image_pixels': image_inputs['pixel_values'],
            'label': torch.tensor(label, dtype=torch.long)
        }


# --- 主执行块 ---
if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content'], inplace=True)
    df['label'] = df['is_recommended']

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    clip_processor = CLIPProcessor.from_pretrained(CONFIG["clip_model"])
    nlp = spacy.load("en_core_web_sm")
    object_detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    train_dataset = SpamCorrelationDataset(train_df, clip_processor, nlp, object_detector)
    val_dataset = SpamCorrelationDataset(val_df, clip_processor, nlp, object_detector)
    test_dataset = SpamCorrelationDataset(test_df, clip_processor, nlp, object_detector)


    def collate_fn(batch):
        # 动态填充，使一个batch内的序列长度一致
        max_nouns = max(item['text_tokens'].shape[0] for item in batch)
        max_crops = max(item['image_pixels'].shape[0] for item in batch)

        text_tokens_padded = []
        image_pixels_padded = []

        for item in batch:
            # 填充文本
            pad_len = max_nouns - item['text_tokens'].shape[0]
            padded = F.pad(item['text_tokens'], (0, 0, 0, pad_len), value=clip_processor.tokenizer.pad_token_id)
            text_tokens_padded.append(padded)

            # 填充图像
            pad_len = max_crops - item['image_pixels'].shape[0]
            padded = F.pad(item['image_pixels'], (0, 0, 0, 0, 0, 0, 0, pad_len))
            image_pixels_padded.append(padded)

        return {
            'text_tokens': torch.stack(text_tokens_padded),
            'image_pixels': torch.stack(image_pixels_padded),
            'labels': torch.stack([item['label'] for item in batch])
        }


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = C3N(CONFIG).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            optimizer.zero_grad()
            logits = model(
                text_tokens=batch['text_tokens'].to(CONFIG["device"]),
                image_pixels=batch['image_pixels'].to(CONFIG["device"])
            )
            loss = loss_fn(logits, batch['labels'].to(CONFIG["device"]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    text_tokens=batch['text_tokens'].to(CONFIG["device"]),
                    image_pixels=batch['image_pixels'].to(CONFIG["device"])
                )
                val_correct += (logits.argmax(1) == batch['labels'].to(CONFIG["device"])).sum().item()

        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch + 1}: Train Loss={(total_loss / len(train_loader)):.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])

    print("\nTesting...")
    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                text_tokens=batch['text_tokens'].to(CONFIG["device"]),
                image_pixels=batch['image_pixels'].to(CONFIG["device"])
            )
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)
    print(f"Test Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")