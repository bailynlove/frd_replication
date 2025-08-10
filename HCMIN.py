import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import AdamW
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse

warnings.filterwarnings("ignore")

# --- 命令行参数 ---
parser = argparse.ArgumentParser(description='Train HCMIN Model')
parser.add_argument('--batch_size', '-bs', type=int, default=16, help='Batch size for training.')
parser.add_argument('--num_epochs', '-ne', type=int, default=3, help='Batch size for training.')
args = parser.parse_args()

# --- 配置 ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "image_dir": "../spams_detection/spam_datasets/crawler/LA/images/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": args.num_epochs,
    "learning_rate": 5e-5,  # Transformer-based models prefer smaller LR
    "bert_model": "bert-base-uncased",
    "shared_dim": 128,  # 论文中提到的共享语义空间维度
    "num_heads": 8,
    "num_classes": 2,
    "contrastive_temp": 0.1,  # 对比学习的温度参数
    "lambda1_feat_cl": 0.2,  # 特征层面CL的权重
    "lambda2_decision_kl": 0.3,  # 决策层面KL散度的权重
    "best_model_path": "hcmin_best_model.pth"
}

print(f"使用设备: {CONFIG['device']}")


# --- 模型子模块 ---
class UnimodalEnhancementModule(nn.Module):
    """模态内增强模块"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class CoAttentionModule(nn.Module):
    """模态间引导协同注意力模块"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value, mask=None):
        attn_output, _ = self.attention(query, key_value, key_value, key_padding_mask=mask)
        query = self.norm1(query + attn_output)
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        return query


# --- 主模型 HCMIN ---
class HCMIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 基础编码器
        self.bert = BertModel.from_pretrained(config["bert_model"])
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的 avgpool 和 fc

        # 投影层
        self.text_proj = nn.Linear(self.bert.config.hidden_size, config["shared_dim"])
        self.image_proj = nn.Linear(resnet.fc.in_features, config["shared_dim"])

        # 双重增强协同注意力模块
        self.text_enhancement = UnimodalEnhancementModule(config["shared_dim"], config["num_heads"])
        self.image_enhancement = UnimodalEnhancementModule(config["shared_dim"], config["num_heads"])

        self.text_guided_co_attention = CoAttentionModule(config["shared_dim"], config["num_heads"])
        self.image_guided_co_attention = CoAttentionModule(config["shared_dim"], config["num_heads"])

        # 双分支分类器
        self.text_classifier = nn.Linear(config["shared_dim"] * 2, config["num_classes"])
        self.image_classifier = nn.Linear(config["shared_dim"] * 2, config["num_classes"])

    def forward(self, text_ids, text_mask, images):
        # 1. 初始特征提取和投影
        text_feat = self.bert(input_ids=text_ids, attention_mask=text_mask).last_hidden_state
        text_feat = self.text_proj(text_feat)  # (B, SeqLen, Dim)

        image_feat_map = self.resnet(images)  # (B, C, H, W)
        B, C, H, W = image_feat_map.shape
        image_feat = image_feat_map.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        image_feat = self.image_proj(image_feat)  # (B, Patches, Dim)

        # 2. 双重增强协同注意力
        # a. 模态内增强
        # 对于文本，我们需要传递 mask 来忽略 padding tokens
        text_padding_mask = (text_mask == 0)
        enhanced_text = self.text_enhancement(text_feat, mask=text_padding_mask)
        # 图像没有 padding，所以不需要 mask
        enhanced_image = self.image_enhancement(image_feat)

        # b. 模态间引导
        # --- START OF FIX ---
        # 当 key_value 是 enhanced_image 时，它没有padding，所以不传 mask
        fused_text = self.image_guided_co_attention(enhanced_text, enhanced_image)

        # 当 key_value 是 enhanced_text 时，它有padding，所以需要传递 text_padding_mask
        fused_image = self.text_guided_co_attention(enhanced_image, enhanced_text, mask=text_padding_mask)
        # --- END OF FIX ---

        # 3. 特征池化
        # 注意：在池化时，需要考虑文本的padding，避免将padding部分计算在内
        # 我们通过将padding位置的特征置零来实现
        masked_enhanced_text = enhanced_text.masked_fill(text_padding_mask.unsqueeze(-1), 0)
        masked_fused_text = fused_text.masked_fill(text_padding_mask.unsqueeze(-1), 0)

        # 计算有效长度用于求平均
        valid_lengths = text_mask.sum(dim=1, keepdim=True)

        pooled_enhanced_text = masked_enhanced_text.sum(dim=1) / valid_lengths
        pooled_fused_text = masked_fused_text.sum(dim=1) / valid_lengths

        # 图像没有padding，直接求平均
        pooled_enhanced_image = enhanced_image.mean(dim=1)
        pooled_fused_image = fused_image.mean(dim=1)

        # 4. 双分支分类
        text_branch_input = torch.cat([pooled_enhanced_text, pooled_fused_text], dim=1)
        image_branch_input = torch.cat([pooled_enhanced_image, pooled_fused_image], dim=1)

        text_logits = self.text_classifier(text_branch_input)
        image_logits = self.image_classifier(image_branch_input)

        return text_logits, image_logits, pooled_enhanced_text, pooled_enhanced_image


# --- 对比损失函数 ---
def contrastive_loss(features1, features2, labels, temp):
    # features1 和 features2 来自不同模态，但标签相同
    sim_matrix = F.cosine_similarity(features1.unsqueeze(1), features2.unsqueeze(0), dim=2) / temp

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features1.device)

    # 移除对角线上的自身匹配
    mask = mask.fill_diagonal_(0)

    log_prob = sim_matrix - torch.log(torch.exp(sim_matrix).sum(1, keepdim=True))

    # 只计算正样本对的损失
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    return -mean_log_prob_pos.mean()


# --- 数据集 ---
class SpamDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor):
        self.df, self.tokenizer, self.image_processor = df, tokenizer, image_processor

    def __len__(self):
        return len(self.df)

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

        return {
            'text': text, 'image': image, 'label': torch.tensor(label, dtype=torch.long)
        }


# --- 主执行块 ---
if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content'], inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    image_processor = lambda x: torch.stack([
        torch.tensor(np.array(img.resize((224, 224)))).permute(2, 0, 1) / 255.0
        for img in x
    ]).float()

    train_dataset = SpamDataset(train_df, tokenizer, image_processor)
    val_dataset = SpamDataset(val_df, tokenizer, image_processor)
    test_dataset = SpamDataset(test_df, tokenizer, image_processor)


    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        images = [item['image'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        image_inputs = image_processor(images)
        return {
            'text_ids': text_inputs['input_ids'], 'text_mask': text_inputs['attention_mask'],
            'images': image_inputs, 'labels': labels
        }


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = HCMIN(CONFIG).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            optimizer.zero_grad()
            text_ids = batch['text_ids'].to(CONFIG["device"])
            text_mask = batch['text_mask'].to(CONFIG["device"])
            images = batch['images'].to(CONFIG["device"])
            labels = batch['labels'].to(CONFIG["device"])

            text_logits, image_logits, text_feat, image_feat = model(text_ids, text_mask, images)

            # 1. 分类损失
            loss_cls_text = F.cross_entropy(text_logits, labels)
            loss_cls_image = F.cross_entropy(image_logits, labels)
            loss_cls = (loss_cls_text + loss_cls_image) / 2

            # 2. 特征层面对比损失
            loss_intra_text = contrastive_loss(text_feat, text_feat, labels, CONFIG["contrastive_temp"])
            loss_intra_image = contrastive_loss(image_feat, image_feat, labels, CONFIG["contrastive_temp"])
            loss_inter = contrastive_loss(text_feat, image_feat, labels, CONFIG["contrastive_temp"])
            loss_feat_cl = loss_intra_text + loss_intra_image + loss_inter

            # 3. 决策层面对比损失 (KL散度)
            prob_text = F.softmax(text_logits, dim=1)
            prob_image = F.softmax(image_logits, dim=1)
            loss_kl = (F.kl_div(prob_text.log(), prob_image, reduction='batchmean') +
                       F.kl_div(prob_image.log(), prob_text, reduction='batchmean')) / 2

            # 组合总损失
            loss = loss_cls + CONFIG["lambda1_feat_cl"] * loss_feat_cl + CONFIG["lambda2_decision_kl"] * loss_kl

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch['text_ids'].to(CONFIG["device"])
                text_mask = batch['text_mask'].to(CONFIG["device"])
                images = batch['images'].to(CONFIG["device"])
                labels = batch['labels'].to(CONFIG["device"])

                text_logits, image_logits, _, _ = model(text_ids, text_mask, images)
                final_probs = (F.softmax(text_logits, dim=1) + F.softmax(image_logits, dim=1)) / 2
                val_correct += (final_probs.argmax(1) == labels).sum().item()

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
            text_ids = batch['text_ids'].to(CONFIG["device"])
            text_mask = batch['text_mask'].to(CONFIG["device"])
            images = batch['images'].to(CONFIG["device"])
            labels = batch['labels'].to(CONFIG["device"])

            text_logits, image_logits, _, _ = model(text_ids, text_mask, images)
            final_probs = (F.softmax(text_logits, dim=1) + F.softmax(image_logits, dim=1)) / 2

            all_preds.extend(final_probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)
    print(f"Test Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")