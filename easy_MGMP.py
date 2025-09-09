import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import random


# --- Configuration ---
class Config:
    DATA_PATH = '../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Hyperparameters
    EMBED_DIM = 128
    MAX_SENT_LEN = 100
    CNN_KERNEL_SIZES = [3, 4, 5]
    CNN_OUT_CHANNELS = 64
    ATTN_HEADS = 4
    META_PATHS_SAMPLES = 10
    DROPOUT = 0.5

    # Training Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 32  # 调整为您显存合适的批次大小
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5


print(f"Using device: {Config.DEVICE}")


# --- 1. Data Preprocessing and Graph Construction ---
class MGMPDataset:
    def __init__(self, config):
        self.config = config
        self.df = self._load_and_clean_data(config.DATA_PATH)
        self._build_vocab_and_mappings()
        self._build_graphs()

    def _load_and_clean_data(self, path):
        print("Loading and cleaning data...")
        df = pd.read_csv(path)
        df = df[df['is_recommended'].isin([0, 1])].copy()
        df['label'] = 1 - df['is_recommended'].astype(int)
        df['content'] = df['content'].fillna('').astype(str)
        df.reset_index(drop=True, inplace=True)
        df['news_id'] = df.index
        return df

    def _build_vocab_and_mappings(self):
        print("Building vocabulary and mappings...")
        self.users = self.df['author_id'].unique()
        self.sources = self.df['biz_alias'].unique()
        self.news = self.df['news_id'].unique()
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.source_to_idx = {s: i for i, s in enumerate(self.sources)}
        self.num_users = len(self.users)
        self.num_sources = len(self.sources)
        self.num_news = len(self.news)
        all_tokens = [token for content in self.df['content'] for token in word_tokenize(content.lower())]
        vocab = sorted(list(set(all_tokens)))
        self.word_to_idx = {w: i + 1 for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab) + 1
        self.word_news_map = defaultdict(list)
        self.news_word_map = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Building word/news maps"):
            tokens = word_tokenize(row['content'].lower())
            word_indices = [self.word_to_idx[t] for t in tokens if t in self.word_to_idx]
            self.news_word_map.append(word_indices)
            for token in set(tokens):
                if token in self.word_to_idx:
                    self.word_news_map[self.word_to_idx[token]].append(row['news_id'])
        print(f"Nodes: {self.num_news} News, {self.num_users} Users, {self.num_sources} Sources.")
        print(f"Vocabulary size: {self.vocab_size}")

    def _build_graphs(self):
        print("Building heterogeneous graph and meta-paths...")
        self.news_user_adj = defaultdict(lambda: -1)  # 使用-1作为默认值
        self.user_news_adj = defaultdict(list)
        self.news_source_adj = defaultdict(lambda: -1)
        self.source_news_adj = defaultdict(list)
        for _, row in self.df.iterrows():
            n_id = row['news_id']
            u_idx = self.user_to_idx[row['author_id']]
            s_idx = self.source_to_idx[row['biz_alias']]
            self.news_user_adj[n_id] = u_idx
            self.user_news_adj[u_idx].append(n_id)
            self.news_source_adj[n_id] = s_idx
            self.source_news_adj[s_idx].append(n_id)

    def get_train_val_test_split(self):
        indices = self.df.index.values
        labels = self.df['label'].values
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, stratify=labels[train_idx], random_state=42)
        return train_idx, val_idx, test_idx


# --- 2. Model Architecture ---
# 辅助模块
class CoarseGrainedEncoder(nn.Module):
    def __init__(self, embed_dim, config):
        super().__init__()
        self.cnn = nn.ModuleList([
            nn.Conv1d(embed_dim, config.CNN_OUT_CHANNELS, k) for k in config.CNN_KERNEL_SIZES
        ])
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(len(config.CNN_KERNEL_SIZES) * config.CNN_OUT_CHANNELS, embed_dim)

    def forward(self, doc_word_embeds):
        x = doc_word_embeds.permute(0, 2, 1)
        conv_outputs = [F.relu(conv(x)) for conv in self.cnn]
        pooled_outputs = [F.max_pool1d(out, out.shape[2]).squeeze(2) for out in conv_outputs]
        x = torch.cat(pooled_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class MetaPathInteractionModule(nn.Module):
    def __init__(self, embed_dim, config):
        super().__init__()
        self.path_encoder = nn.GRU(embed_dim, embed_dim, num_layers=1, batch_first=True)
        self.mhsa = nn.MultiheadAttention(embed_dim, config.ATTN_HEADS, batch_first=True, dropout=config.DROPOUT)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, path_instances_embeds):
        batch_size, num_samples, path_len, dim = path_instances_embeds.shape
        path_instances_flat = path_instances_embeds.view(batch_size * num_samples, path_len, dim)
        _, hidden_state = self.path_encoder(path_instances_flat)
        encoded_paths = hidden_state.squeeze(0).view(batch_size, num_samples, dim)
        attn_output, _ = self.mhsa(encoded_paths, encoded_paths, encoded_paths)
        aggregated_embeds = self.norm(encoded_paths + attn_output)
        return aggregated_embeds.mean(dim=1)


# 主模型
class MGMP(nn.Module):
    def __init__(self, data_handler, config):
        super().__init__()
        self.config = config
        self.data_handler = data_handler
        self.news_embeds = nn.Embedding(data_handler.num_news, config.EMBED_DIM)
        self.user_embeds = nn.Embedding(data_handler.num_users, config.EMBED_DIM)
        self.source_embeds = nn.Embedding(data_handler.num_sources, config.EMBED_DIM)
        self.word_embeds = nn.Embedding(data_handler.vocab_size, config.EMBED_DIM, padding_idx=0)
        self.word_attention = nn.MultiheadAttention(config.EMBED_DIM, config.ATTN_HEADS, batch_first=True)
        self.coarse_encoder = CoarseGrainedEncoder(config.EMBED_DIM, config)
        self.interaction_nsn = MetaPathInteractionModule(config.EMBED_DIM, config)
        self.interaction_nun = MetaPathInteractionModule(config.EMBED_DIM, config)
        self.meta_path_attention = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.EMBED_DIM // 2),
            nn.Tanh(),
            nn.Linear(config.EMBED_DIM // 2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.EMBED_DIM, 2)
        )

    def _get_fine_grained_word_embeds(self, word_indices):
        word_embeds_list = []
        for word_idx in word_indices:
            if word_idx == 0: continue  # Skip padding
            news_context_ids = self.data_handler.word_news_map.get(word_idx.item(), [])
            if not news_context_ids:
                word_embeds_list.append(self.word_embeds(word_idx))
                continue

            news_context_ids = torch.LongTensor(random.sample(news_context_ids, k=min(len(news_context_ids), 50))).to(
                self.config.DEVICE)
            context_news_embeds = self.news_embeds(news_context_ids).unsqueeze(0)
            query = self.word_embeds(word_idx).unsqueeze(0).unsqueeze(0)
            attn_output, _ = self.word_attention(query, context_news_embeds, context_news_embeds)
            word_embeds_list.append(attn_output.squeeze(0).squeeze(0))

        return torch.stack(word_embeds_list) if word_embeds_list else torch.empty(0, self.config.EMBED_DIM).to(
            self.config.DEVICE)

    def _sample_meta_paths_for_batch(self, news_ids, path_type):
        batch_instances = []
        for n_id in news_ids:
            node_instances = []
            if path_type == 'NSN':
                adj1, adj2, embed1, embed2 = self.data_handler.news_source_adj, self.data_handler.source_news_adj, self.source_embeds, self.news_embeds
            else:  # NUN
                adj1, adj2, embed1, embed2 = self.data_handler.news_user_adj, self.data_handler.user_news_adj, self.user_embeds, self.news_embeds

            for _ in range(self.config.META_PATHS_SAMPLES):
                neighbor1_idx = adj1[n_id.item()]
                if neighbor1_idx == -1 or not adj2.get(neighbor1_idx) or len(adj2[neighbor1_idx]) < 2:
                    node_instances.append(torch.zeros(3, self.config.EMBED_DIM).to(self.config.DEVICE))
                    continue

                end_node_idx = random.choice([n for n in adj2[neighbor1_idx] if n != n_id.item()])
                start_embed = self.news_embeds(n_id)
                n1_embed = embed1(torch.LongTensor([neighbor1_idx]).to(self.config.DEVICE)).squeeze(0)
                end_embed = embed2(torch.LongTensor([end_node_idx]).to(self.config.DEVICE)).squeeze(0)
                node_instances.append(torch.stack([start_embed, n1_embed, end_embed]))

            batch_instances.append(torch.stack(node_instances))

        return torch.stack(batch_instances)

    def forward(self, news_ids, news_word_indices):
        unique_word_indices = torch.unique(news_word_indices.flatten())
        unique_word_indices_no_pad = unique_word_indices[unique_word_indices != 0]

        if unique_word_indices_no_pad.numel() > 0:
            fine_grained_embeds = self._get_fine_grained_word_embeds(unique_word_indices_no_pad)
            temp_word_embed_table = self.word_embeds.weight.clone()
            temp_word_embed_table[unique_word_indices_no_pad] = fine_grained_embeds
            doc_word_embeds = F.embedding(news_word_indices, temp_word_embed_table, padding_idx=0)
        else:
            doc_word_embeds = self.word_embeds(news_word_indices)

        semantic_embeds = self.coarse_encoder(doc_word_embeds)

        nsn_instances = self._sample_meta_paths_for_batch(news_ids, 'NSN')
        nun_instances = self._sample_meta_paths_for_batch(news_ids, 'NUN')
        nsn_embeds = self.interaction_nsn(nsn_instances)
        nun_embeds = self.interaction_nun(nun_instances)

        meta_path_stack = torch.stack([nsn_embeds, nun_embeds], dim=1)
        attn_weights = F.softmax(self.meta_path_attention(meta_path_stack), dim=1)
        aggregated_structural_embeds = (meta_path_stack * attn_weights).sum(dim=1)

        combined_embeds = torch.cat([semantic_embeds, aggregated_structural_embeds], dim=1)
        logits = self.classifier(combined_embeds)

        return logits


# --- 3. Dataset, Training, and Evaluation ---
class NewsDataset(Dataset):
    def __init__(self, df, indices, news_word_map, max_len):
        self.labels = df.iloc[indices]['label'].values
        self.indices = indices
        self.news_sequences = []
        for idx in indices:
            seq = news_word_map[idx]
            seq = seq[:max_len] if len(seq) > max_len else seq + [0] * (max_len - len(seq))
            self.news_sequences.append(torch.LongTensor(seq))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx], self.news_sequences[idx]


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for news_ids, labels, news_sequences in tqdm(data_loader, desc="Training"):
        news_ids, labels, news_sequences = news_ids.to(device), labels.to(device), news_sequences.to(device)
        optimizer.zero_grad()
        logits = model(news_ids, news_sequences)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for news_ids, labels, news_sequences in tqdm(data_loader, desc="Evaluating"):
            news_ids, labels, news_sequences = news_ids.to(device), labels.to(device), news_sequences.to(device)
            logits = model(news_ids, news_sequences)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return acc, precision, recall, f1


def main():
    config = Config()

    data_handler = MGMPDataset(config)
    train_idx, val_idx, test_idx = data_handler.get_train_val_test_split()

    train_dataset = NewsDataset(data_handler.df, train_idx, data_handler.news_word_map, config.MAX_SENT_LEN)
    val_dataset = NewsDataset(data_handler.df, val_idx, data_handler.news_word_map, config.MAX_SENT_LEN)
    test_dataset = NewsDataset(data_handler.df, test_idx, data_handler.news_word_map, config.MAX_SENT_LEN)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    model = MGMP(data_handler, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting Training ---")
    for epoch in range(config.EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, config.DEVICE)
        val_acc, val_pre, val_rec, val_f1 = evaluate(model, val_loader, config.DEVICE)
        print(f"Epoch {epoch + 1:02d}/{config.EPOCHS} | Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Pre: {val_pre:.4f} | "
              f"Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")

    print("\n--- Final Evaluation on Test Set ---")
    test_acc, test_pre, test_rec, test_f1 = evaluate(model, test_loader, config.DEVICE)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_pre:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")


if __name__ == '__main__':
    main()