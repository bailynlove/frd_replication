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
# 导入PyTorch Geometric (PyG) 和 sentence-transformers
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

# --- 命令行参数设置 ---
parser = argparse.ArgumentParser(description='Train CMGN Model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
args = parser.parse_args()

# --- 改进点 5: 对齐超参数 ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_model": "bert-base-uncased",
    "st_model": "sentence-transformers/all-MiniLM-L6-v2",  # 用于GNN节点特征
    "embedding_dim": 128,
    "fusion_dim": 64,
    "tokens_mlp_dim": 512,
    "channels_mlp_dim": 128,
    "num_classes": 2,
    "epochs": 20,
    "learning_rate": 1e-3,
    "best_model_path": "cmgn_best_model_v2.pth"
}

print(f"使用设备: {CONFIG['device']}")
print(f"批处理大小: {args.batch_size}")


# --- 改进点 2: 忠实复现 RWKV MLP-Mixer ---
class RWKV_MLP_Mixer(nn.Module):
    def __init__(self, dim, num_patches, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, dim)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x)
        x = x + self.channel_mix(x)
        return x


# --- 改进点 3: 实现真正的 Text GNN 模块 ---
class TextGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, graph_batch):
        x = self.conv1(graph_batch.x, graph_batch.edge_index)
        x = F.relu(x)
        x = self.conv2(x, graph_batch.edge_index)
        # 使用均值池化得到每个图的表示
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(x, graph_batch.batch)


# --- 主模型 CMGN ---
class CMGN(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.embedding_dim = config["embedding_dim"]

        self.news_text_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.mixer = RWKV_MLP_Mixer(self.embedding_dim, 128, config["tokens_mlp_dim"], config["channels_mlp_dim"])

        self.st_model = SentenceTransformer(config["st_model"], device=config['device'])
        gnn_input_dim = self.st_model.get_sentence_embedding_dimension()
        self.text_gnn = TextGNN(gnn_input_dim, self.embedding_dim * 2, self.embedding_dim)

        self.numerical_lstm = nn.LSTM(input_size=1, hidden_size=self.embedding_dim, batch_first=True)

        # --- 改进点 4: 精确实现交叉特征融合 ---
        self.mixer_proj = nn.Linear(self.embedding_dim, config["fusion_dim"])
        self.gnn_proj = nn.Linear(self.embedding_dim, config["fusion_dim"])
        self.fusion_proj = nn.Linear(config["fusion_dim"] * config["fusion_dim"], self.embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.embedding_dim, config["num_classes"])
        )

    def forward(self, news_text_ids, additional_text_list, numerical_data):
        news_embed = self.news_text_embedding(news_text_ids)
        mixer_out = self.mixer(news_embed).mean(dim=1)

        # --- GNN动态建图和处理 ---
        graph_list = []
        for texts in additional_text_list:
            if not texts:  # 处理没有附加文本的情况
                texts = ["<pad>"]
            with torch.no_grad():
                node_features = self.st_model.encode(texts, convert_to_tensor=True)
            num_nodes = node_features.shape[0]
            if num_nodes > 1:
                sim_matrix = util.cos_sim(node_features, node_features)
                # k-NN建图 (k=2)
                _, top_k_indices = torch.topk(sim_matrix, k=min(3, num_nodes), dim=1)
                edge_list = []
                for i in range(num_nodes):
                    for j in top_k_indices[i]:
                        if i != j:
                            edge_list.append([i, j.item()])
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            graph_list.append(Data(x=node_features, edge_index=edge_index.to(self.config['device'])))

        graph_batch = Batch.from_data_list(graph_list)
        gnn_out = self.text_gnn(graph_batch)

        _, (lstm_out, _) = self.numerical_lstm(numerical_data.unsqueeze(-1))
        lstm_out = lstm_out.squeeze(0)

        # --- 交叉特征融合 ---
        m_prime = self.mixer_proj(mixer_out)  # (B, fusion_dim)
        g_prime = self.gnn_proj(gnn_out)  # (B, fusion_dim)

        # 外积操作
        fused_outer = torch.bmm(m_prime.unsqueeze(2), g_prime.unsqueeze(1))  # (B, fusion_dim, fusion_dim)
        fused_flat = fused_outer.view(fused_outer.size(0), -1)  # (B, fusion_dim * fusion_dim)
        fused_features = self.fusion_proj(fused_flat)  # (B, embedding_dim)

        final_features = torch.cat([fused_features, lstm_out], dim=1)
        logits = self.classifier(final_features)
        return logits


# --- Focal Loss (Unchanged) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


# --- 数据集 (Unchanged) ---
class SpamDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.additional_text_cols = ['author_name', 'author_loc', 'biz_name', 'biz_categories']
        self.numerical_cols = ['author_friend_sum', 'author_review_sum', 'biz_reviewCount', 'biz_rating']

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        news_text = str(row['content'])
        additional_texts = [f"{col}: {str(row[col])}" for col in self.additional_text_cols if
                            pd.notna(row[col]) and str(row[col])]
        numerical_data = [float(row.get(col, 0.0)) for col in self.numerical_cols]
        # 归一化数值数据
        numerical_data = (np.array(numerical_data) - np.mean(numerical_data)) / (np.std(numerical_data) + 1e-6)
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
                news_text_ids=batch['news_text_ids'].to(CONFIG["device"]),
                additional_text_list=batch['additional_texts'],
                numerical_data=batch['numerical_data'].to(CONFIG["device"])
            )
            loss = loss_fn(logits, batch['labels'].to(CONFIG["device"]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == batch['labels'].to(CONFIG["device"])).sum().item()

        train_acc, train_loss = total_correct / len(train_dataset), total_loss / len(train_loader)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} Validation"):
                logits = model(
                    news_text_ids=batch['news_text_ids'].to(CONFIG["device"]),
                    additional_text_list=batch['additional_texts'],
                    numerical_data=batch['numerical_data'].to(CONFIG["device"])
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
                news_text_ids=batch['news_text_ids'].to(CONFIG["device"]),
                additional_text_list=batch['additional_texts'],
                numerical_data=batch['numerical_data'].to(CONFIG["device"])
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