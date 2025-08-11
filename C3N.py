import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
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
    "epochs": 0,
    "learning_rate": 1e-5,
    "clip_model": "openai/clip-vit-base-patch32",
    "num_nouns": 20,
    "num_crops": 5,
    "hidden_dim": 128,
    "num_classes": 2,
    "best_model_path": "c3n_best_model_v2.pth"
}

print(f"使用设备: {CONFIG['device']}")


# --- 主模型 C3N (修正CNN初始化) ---
class C3N(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.clip = CLIPModel.from_pretrained(config["clip_model"])
        self.enhancement_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.clip.config.projection_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        # --- 修正点: 确保 in_channels 与固定输入匹配 ---
        self.correlation_cnn = nn.Conv1d(
            in_channels=config["num_crops"] + 1,  # M = num_crops + 1 (全图)
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        fusion_input_dim = self.clip.config.projection_dim * 2 + 64
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(config["hidden_dim"], config["num_classes"])

    def forward(self, text_tokens, image_pixels):
        B, N, SeqLen = text_tokens.shape
        _, M, C, H, W = image_pixels.shape

        text_features = self.clip.get_text_features(input_ids=text_tokens.view(-1, SeqLen)).view(B, N, -1)
        image_features = self.clip.get_image_features(pixel_values=image_pixels.view(-1, C, H, W)).view(B, M, -1)

        enhanced_text = self.enhancement_transformer(text_features)
        enhanced_image = self.enhancement_transformer(image_features)

        sim_matrix = F.cosine_similarity(enhanced_text.unsqueeze(2), enhanced_image.unsqueeze(1), dim=-1)

        correlation_features = self.correlation_cnn(sim_matrix.permute(0, 2, 1)).mean(dim=2)

        global_text_feat = enhanced_text[:, 0, :]
        global_image_feat = enhanced_image[:, 0, :]

        fused_features = self.fusion_mlp(torch.cat([global_text_feat, global_image_feat, correlation_features], dim=1))

        logits = self.classifier(fused_features)
        return logits


# --- 数据集 (修正以返回固定长度的列表) ---
class SpamCorrelationDataset(Dataset):
    def __init__(self, df, clip_processor, nlp, object_detector):
        self.df, self.clip_processor, self.nlp, self.object_detector = df, clip_processor, nlp, object_detector
        self.object_detector.to(CONFIG['device']).eval()

    def __len__(self):
        return len(self.df)

    def _extract_nouns(self, text):
        doc = self.nlp(text)
        nouns = [token.text for token in doc if token.pos_ == 'NOUN']
        # --- 修正点: 填充或截断到固定长度 ---
        padded_nouns = nouns[:CONFIG['num_nouns']]
        padded_nouns += [''] * (CONFIG['num_nouns'] - len(padded_nouns))
        return padded_nouns

    def _extract_crops(self, image):
        from torchvision.transforms.functional import to_tensor
        img_tensor = to_tensor(image).to(CONFIG['device'])
        with torch.no_grad():
            predictions = self.object_detector([img_tensor])[0]

        top_indices = torch.topk(predictions['scores'], k=min(len(predictions['scores']), CONFIG['num_crops'])).indices
        boxes = predictions['boxes'][top_indices].cpu().numpy().astype(int)

        crops = [image.crop(tuple(box)) for box in boxes]
        # --- 修正点: 填充或截断到固定长度 ---
        padded_crops = crops[:CONFIG['num_crops']]
        # 使用空白图像填充
        padded_crops += [Image.new('RGB', (224, 224))] * (CONFIG['num_crops'] - len(padded_crops))
        return padded_crops

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

        nouns = self._extract_nouns(text)
        crops = self._extract_crops(image)

        text_list = [text] + nouns
        image_list = [image] + crops

        text_inputs = self.clip_processor(text=text_list, return_tensors='pt', padding='max_length', truncation=True,
                                          max_length=77)
        image_inputs = self.clip_processor(images=image_list, return_tensors='pt', padding=True)

        return {
            'text_tokens': text_inputs['input_ids'],
            'image_pixels': image_inputs['pixel_values'],
            'label': torch.tensor(label, dtype=torch.long)
        }


# --- 主执行块 (修正collate_fn) ---
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


    # --- 修正点: 简化 collate_fn ---
    def collate_fn(batch):
        return {
            'text_tokens': torch.stack([item['text_tokens'] for item in batch]),
            'image_pixels': torch.stack([item['image_pixels'] for item in batch]),
            'labels': torch.stack([item['label'] for item in batch])
        }


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

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