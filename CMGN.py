import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
# 导入PyTorch Geometric (PyG)
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# --- 命令行参数设置 ---
parser = argparse.ArgumentParser(description='Train CMGN Model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
args = parser.parse_args()

# --- 配置 ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_model": "bert-base-uncased",
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_classes": 2,
    "epochs": 20,
    "learning_rate": 1e-3,
    "best_model_path": "cmgn_best_model.pth"
}

print(f"使用设备: {CONFIG['device']}")
print(f"批处理大小: {args.batch_size}")


# --- 模型子模块 ---

class RWKV_MLP_Block(nn.Module):
    """简化的MLP-Mixer块，用于处理序列数据"""

    def __init__(self, dim, num_patches):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(num_patches, num_patches),
            nn.GELU()
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x shape: (batch, num_patches, dim)
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(x)
        return x


class TextGNN(nn.Module):
    """处理附加文本的图神经网络"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, text_list, tokenizer, device):
        # 1. 为每个batch动态构建图
        batch_graphs = []
        for texts in text_list:
            # 使用BERT tokenizer获取词嵌入
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
            # 使用简单的词嵌入均值作为节点特征
            with torch.no_grad():
                # 这是一个简化的嵌入获取方式，更复杂的可以用BERT模型
                node_features = tokenizer.get_vocab()[inputs['input_ids']].float().mean(dim=1)

            num_nodes = len(texts)
            # 构建全连接图（或k-NN图）
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)

            graph_data = Data(x=node_features, edge_index=edge_index)
            batch_graphs.append(graph_data)

        # PyG的DataLoader可以自动处理batching，但这里为了整合，我们手动处理
        # 简化：对每个图单独处理然后池化
        graph_outputs = []
        for graph in batch_graphs:
            x, edge_index = graph.x, graph.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            # 使用均值池化得到整个图的表示
            graph_outputs.append(x.mean(dim=0))

        return torch.stack(graph_outputs)


# --- 主模型 CMGN ---
class CMGN(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.embedding_dim = config["embedding_dim"]

        # 模块1: 新闻文本处理 (RWKV MLP-Mixer)
        self.news_text_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.mixer_blocks = nn.Sequential(*[RWKV_MLP_Block(self.embedding_dim, 128) for _ in range(4)])  # 假设最大长度128

        # 模块2: 附加文本处理 (Text GNN)
        # GNN的输入维度是tokenizer词汇表大小
        self.text_gnn = TextGNN(vocab_size, config["hidden_dim"], self.embedding_dim)

        # 模块3: 数值数据处理 (LSTM)
        self.numerical_lstm = nn.LSTM(input_size=1, hidden_size=self.embedding_dim, batch_first=True)

        # 模块4: 交叉特征融合
        self.fusion_norm1 = nn.LayerNorm(self.embedding_dim)
        self.fusion_norm2 = nn.LayerNorm(self.embedding_dim)

        # 分类器
        # 输入维度 = mixer输出 + gnn输出 + lstm输出
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config["hidden_dim"], config["num_classes"])
        )

    def forward(self, news_text, additional_text_list, numerical_data, tokenizer):
        # 1. 处理新闻文本
        news_embed = self.news_text_embedding(news_text)  # (B, SeqLen) -> (B, SeqLen, EmbDim)
        mixer_out = self.mixer_blocks(news_embed).mean(dim=1)  # (B, EmbDim)

        # 2. 处理附加文本 (由于GNN的复杂性，这里简化处理)
        # 在真实场景中，GNN的处理会更复杂。这里我们用一个简化的TextGNN
        # gnn_out = self.text_gnn(additional_text_list, tokenizer, self.config['device'])
        # 简化：由于动态建图和batching的复杂性，我们先用一个简单的嵌入均值代替
        gnn_out_list = []
        for texts in additional_text_list:
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.config['device'])
            embeds = self.news_text_embedding(inputs['input_ids'])
            gnn_out_list.append(embeds.mean(dim=(0, 1)))
        gnn_out = torch.stack(gnn_out_list)

        # 3. 处理数值数据
        # (B, NumFeats) -> (B, NumFeats, 1)
        numerical_data = numerical_data.unsqueeze(-1)
        _, (lstm_out, _) = self.numerical_lstm(numerical_data)
        lstm_out = lstm_out.squeeze(0)  # (B, EmbDim)

        # 4. 交叉特征融合 (Mixer_out 和 GNN_out)
        mixer_norm = self.fusion_norm1(mixer_out)
        gnn_norm = self.fusion_norm2(gnn_out)

        # 计算注意力权重 (Eq. 10)
        attention_weights = F.softmax(torch.bmm(mixer_norm.unsqueeze(1), gnn_norm.unsqueeze(2)), dim=-1)  # (B, 1, 1)

        # 融合特征 (Eq. 11的简化实现)
        fused_features = mixer_out + attention_weights.squeeze(-1) * gnn_out

        # 5. 最终特征拼接 (融合特征 + 数值特征)
        final_features = torch.cat([fused_features, lstm_out], dim=1)

        # 6. 分类
        logits = self.classifier(final_features)
        return logits


# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# --- 数据集 ---
class SpamDataset(Dataset):
    def __init__(self, df):
        self.df = df
        # 定义哪些列是附加文本，哪些是数值
        self.additional_text_cols = ['author_name', 'author_loc', 'biz_name', 'biz_categories']
        self.numerical_cols = ['author_friend_sum', 'author_review_sum', 'biz_reviewCount', 'biz_rating']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        news_text = str(row['content'])

        additional_texts = [str(row[col]) for col in self.additional_text_cols if pd.notna(row[col])]
        # 填充缺失的数值数据
        numerical_data = [float(row.get(col, 0.0)) for col in self.numerical_cols]

        label = int(row['is_recommended'])

        return {
            'news_text': news_text,
            'additional_texts': additional_texts,
            'numerical_data': torch.tensor(numerical_data, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }


# --- 主执行块 ---
if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content'], inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])

    train_dataset = SpamDataset(train_df)
    val_dataset = SpamDataset(val_df)
    test_dataset = SpamDataset(test_df)


    def collate_fn(batch):
        news_texts = [item['news_text'] for item in batch]
        additional_texts_list = [item['additional_texts'] for item in batch]
        numerical_data = torch.stack([item['numerical_data'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])

        news_text_tokenized = tokenizer(news_texts, padding='max_length', max_length=128, truncation=True,
                                        return_tensors='pt')

        return {
            'news_text_ids': news_text_tokenized['input_ids'],
            'news_text_mask': news_text_tokenized['attention_mask'],
            'additional_texts': additional_texts_list,
            'numerical_data': numerical_data,
            'labels': labels
        }


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = CMGN(CONFIG, vocab_size=tokenizer.vocab_size).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = FocalLoss()

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss, total_correct = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} Training"):
            optimizer.zero_grad()
            logits = model(
                news_text=batch['news_text_ids'].to(CONFIG["device"]),
                additional_text_list=batch['additional_texts'],
                numerical_data=batch['numerical_data'].to(CONFIG["device"]),
                tokenizer=tokenizer
            )
            loss = loss_fn(logits, batch['labels'].to(CONFIG["device"]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == batch['labels'].to(CONFIG["device"])).sum().item()

        train_acc = total_correct / len(train_dataset)
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} Validation"):
                logits = model(
                    news_text=batch['news_text_ids'].to(CONFIG["device"]),
                    additional_text_list=batch['additional_texts'],
                    numerical_data=batch['numerical_data'].to(CONFIG["device"]),
                    tokenizer=tokenizer
                )
                val_correct += (logits.argmax(1) == batch['labels'].to(CONFIG["device"])).sum().item()

        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    print("\nTraining complete. Loading best model for final testing...")
    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            logits = model(
                news_text=batch['news_text_ids'].to(CONFIG["device"]),
                additional_text_list=batch['additional_texts'],
                numerical_data=batch['numerical_data'].to(CONFIG["device"]),
                tokenizer=tokenizer
            )
            preds = logits.argmax(1)
            all_labels.extend(batch['labels'].numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)

    print("\n--- Test Results ---")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("--------------------")