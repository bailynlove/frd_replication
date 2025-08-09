import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

warnings.filterwarnings("ignore")

# --- 命令行参数 ---
parser = argparse.ArgumentParser(description='Train CoST Model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
args = parser.parse_args()

# --- 配置 ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 20,
    "learning_rate": 1e-4,
    "bert_model": "bert-base-uncased",
    "hidden_dim": 128,
    "num_classes": 2,
    "best_model_path": "cost_best_model.pth"
}

print(f"使用设备: {CONFIG['device']}")


# --- 模型子模块 ---
class GA_LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gat = GATConv(input_dim + hidden_dim, hidden_dim, heads=1, concat=False)
        self.linear_ih = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear_hh = nn.Linear(hidden_dim, 4 * hidden_dim)

    def forward(self, x, h_prev, c_prev, edge_index):
        h_agg = self.gat(torch.cat([x, h_prev], dim=1), edge_index)
        gates = self.linear_ih(h_agg) + self.linear_hh(h_prev)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate, forgetgate, cellgate, outgate = torch.sigmoid(ingate), torch.sigmoid(forgetgate), torch.tanh(
            cellgate), torch.sigmoid(outgate)
        c_next = (forgetgate * c_prev) + (ingate * cellgate)
        h_next = outgate * torch.tanh(c_next)
        return h_next, c_next


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads=4):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        x = self.pos_encoder(x)
        return self.transformer_encoder(x)


class CoST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config["bert_model"])
        bert_dim = self.bert.config.hidden_size
        self.feature_proj = nn.Linear(bert_dim, config["hidden_dim"])
        self.ga_lstm_cell = GA_LSTMCell(config["hidden_dim"], config["hidden_dim"])
        self.temporal_encoder = TemporalEncoder(config["hidden_dim"])
        self.gate = nn.Sequential(nn.Linear(config["hidden_dim"] * 2, config["hidden_dim"]), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"] // 2, config["num_classes"])
        )

    def forward(self, graph_batch):
        with torch.no_grad():
            node_features = self.bert(input_ids=graph_batch.input_ids,
                                      attention_mask=graph_batch.attention_mask).last_hidden_state.mean(dim=1)
        x = self.feature_proj(node_features)

        h = torch.zeros(x.size(0), self.config["hidden_dim"], device=self.config["device"])
        c = torch.zeros(x.size(0), self.config["hidden_dim"], device=self.config["device"])
        h_struct, _ = self.ga_lstm_cell(x, h, c, graph_batch.edge_index)

        from torch_geometric.nn import global_mean_pool
        struct_feat = global_mean_pool(h_struct, graph_batch.batch)

        temporal_inputs = []
        for i in range(graph_batch.num_graphs):
            graph_slice = graph_batch[i]
            nodes_in_graph = x[graph_batch.ptr[i]:graph_batch.ptr[i + 1]]
            if graph_slice.t.numel() > 0 and nodes_in_graph.numel() > 0:
                sorted_indices = torch.argsort(graph_slice.t)
                if sorted_indices.max() < len(nodes_in_graph):
                    temporal_inputs.append(nodes_in_graph[sorted_indices])
                else:
                    temporal_inputs.append(nodes_in_graph)
            else:
                temporal_inputs.append(nodes_in_graph)

        padded_temporal = nn.utils.rnn.pad_sequence(temporal_inputs, batch_first=True)
        if padded_temporal.numel() > 0:
            temp_feat = self.temporal_encoder(padded_temporal).mean(dim=1)
        else:
            temp_feat = torch.zeros_like(struct_feat)

        gate_val = self.gate(torch.cat([struct_feat, temp_feat], dim=1))
        fused_feat = gate_val * struct_feat + (1 - gate_val) * temp_feat

        logits = self.classifier(fused_feat)
        return logits


class SpamPropagationDataset(Dataset):
    def __init__(self, df, tokenizer, user_map, biz_map, adj_matrix):
        self.df, self.tokenizer = df, tokenizer
        self.user_map, self.biz_map, self.adj_matrix = user_map, biz_map, adj_matrix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['is_recommended'])
        user_idx, biz_idx = self.user_map.get(row['author_name'], -1), self.biz_map.get(row['biz_name'], -1)

        node_indices = []
        if user_idx != -1: node_indices.append(user_idx)
        if biz_idx != -1: node_indices.append(biz_idx + len(self.user_map))

        neighbors = set(node_indices)
        for node in node_indices:
            neighbors.update(self.adj_matrix[node].nonzero()[0])

        subgraph_nodes = sorted(list(neighbors))
        if not subgraph_nodes: subgraph_nodes = [user_idx if user_idx != -1 else 0]
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}

        edge_list = []
        for i in subgraph_nodes:
            for j in self.adj_matrix[i].nonzero()[0]:
                if j in subgraph_nodes:
                    edge_list.append([node_map[i], node_map[j]])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0),
                                                                                                              dtype=torch.long)

        node_texts, timestamps = [], []
        for node_idx in subgraph_nodes:
            if node_idx < len(self.user_map):
                user_name = list(self.user_map.keys())[list(self.user_map.values()).index(node_idx)]
                node_texts.append(f"user: {user_name}")
                timestamps.append(pd.to_datetime(row['create_time']).timestamp())
            else:
                biz_name = list(self.biz_map.keys())[list(self.biz_map.values()).index(node_idx - len(self.user_map))]
                node_texts.append(f"business: {biz_name}")
                timestamps.append(pd.to_datetime(row['create_time']).timestamp())

        node_inputs = self.tokenizer(node_texts, padding='max_length', max_length=32, truncation=True,
                                     return_tensors='pt')

        return Data(
            edge_index=edge_index,
            input_ids=node_inputs['input_ids'],
            attention_mask=node_inputs['attention_mask'],
            t=torch.tensor(timestamps, dtype=torch.float),
            y=torch.tensor(label, dtype=torch.long)
        )


if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content', 'author_name', 'biz_name', 'create_time'], inplace=True)

    users, businesses = df['author_name'].unique(), df['biz_name'].unique()
    user_map, biz_map = {name: i for i, name in enumerate(users)}, {name: i for i, name in enumerate(businesses)}
    num_nodes = len(users) + len(businesses)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for _, row in df.iterrows():
        u_idx, b_idx = user_map[row['author_name']], biz_map[row['biz_name']] + len(users)
        adj_matrix[u_idx, b_idx] = adj_matrix[b_idx, u_idx] = 1

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])

    train_dataset = SpamPropagationDataset(train_df, tokenizer, user_map, biz_map, adj_matrix)
    val_dataset = SpamPropagationDataset(val_df, tokenizer, user_map, biz_map, adj_matrix)
    test_dataset = SpamPropagationDataset(test_df, tokenizer, user_map, biz_map, adj_matrix)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Batch.from_data_list)

    model = CoST(CONFIG).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            batch = batch.to(CONFIG["device"])
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(CONFIG["device"])
                logits = model(batch)
                val_correct += (logits.argmax(1) == batch.y).sum().item()

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
            batch = batch.to(CONFIG["device"])
            logits = model(batch)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)
    print(f"Test Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")