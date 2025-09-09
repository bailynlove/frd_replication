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
    EMBED_DIM = 128  # General embedding dimension for nodes and words
    MAX_SENT_LEN = 100  # Max words per news item (for CNN)
    CNN_KERNEL_SIZES = [3, 4, 5]  # Kernel sizes for the CNN
    CNN_OUT_CHANNELS = 64  # Number of filters for each kernel size
    ATTN_HEADS = 4  # Number of heads for multi-head attention
    META_PATHS_SAMPLES = 10  # Number of meta-path instances to sample per node
    DROPOUT = 0.5

    # Training Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 64
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
        df.reset_index(drop=True, inplace=True)  # Important for using index as news_id
        df['news_id'] = df.index
        return df

    def _build_vocab_and_mappings(self):
        print("Building vocabulary and mappings...")
        # Node mappings
        self.users = self.df['author_id'].unique()
        self.sources = self.df['biz_alias'].unique()
        self.news = self.df['news_id'].unique()

        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.source_to_idx = {s: i for i, s in enumerate(self.sources)}
        # News_id is already the index, so mapping is identity

        self.num_users = len(self.users)
        self.num_sources = len(self.sources)
        self.num_news = len(self.news)

        # Word vocabulary and mappings
        all_tokens = [token for content in self.df['content'] for token in word_tokenize(content.lower())]
        vocab = sorted(list(set(all_tokens)))
        self.word_to_idx = {w: i + 1 for i, w in enumerate(vocab)}  # +1 for padding token 0
        self.vocab_size = len(vocab) + 1

        # Word -> News mapping (for fine-grained learning)
        self.word_news_map = defaultdict(list)
        # News -> Word mapping (for coarse-grained learning)
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
        # Adjacency lists for graph structure
        self.news_user_adj = defaultdict(list)
        self.user_news_adj = defaultdict(list)
        self.news_source_adj = defaultdict(list)
        self.source_news_adj = defaultdict(list)

        for _, row in self.df.iterrows():
            n_id = row['news_id']
            u_idx = self.user_to_idx[row['author_id']]
            s_idx = self.source_to_idx[row['biz_alias']]

            self.news_user_adj[n_id].append(u_idx)
            self.user_news_adj[u_idx].append(n_id)
            self.news_source_adj[n_id].append(s_idx)
            self.source_news_adj[s_idx].append(n_id)

    def get_train_val_test_split(self):
        indices = self.df.index.values
        labels = self.df['label'].values
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, stratify=labels[train_idx],
                                              random_state=42)  # 0.1 of original
        return train_idx, val_idx, test_idx


# --- 2. Model Architecture ---

# Module 2a: Multi-granularity Semantic Learning
# Module 2a: Multi-granularity Semantic Learning
class MultiGranularitySemanticModule(nn.Module):
    def __init__(self, num_news, vocab_size, embed_dim, cnn_kernel_sizes, cnn_out_channels, attn_heads, dropout,
                 max_sent_len, device):
        super().__init__()

        # --- 将所有需要的参数保存为模块属性 ---
        self.max_sent_len = max_sent_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device = device

        # Initial news embeddings (randomly initialized as per paper)
        self.news_embeds = nn.Embedding(num_news, embed_dim)

        # Fine-grained: Multi-head attention to learn word embeddings
        self.word_attention = nn.MultiheadAttention(embed_dim, attn_heads, batch_first=True)

        # Coarse-grained: CNN for document embeddings
        self.cnn = nn.ModuleList([
            nn.Conv1d(embed_dim, cnn_out_channels, k) for k in cnn_kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(cnn_kernel_sizes) * cnn_out_channels, embed_dim)

    # 这是修正后的 forward 方法
    def forward(self, all_news_ids, word_news_map, news_word_map):
        # --- Fine-grained Step ---
        # --- 这是关键修改 ---
        # 从模块的属性 (self) 中获取 vocab_size 和 embed_dim，而不是 config
        all_word_embeds = torch.zeros(self.vocab_size, self.embed_dim).to(self.device)

        for word_idx, news_indices in word_news_map.items():
            if not news_indices: continue
            # Get embeddings of all news articles this word appears in
            news_ids_tensor = torch.LongTensor(news_indices).to(self.device)
            context_news_embeds = self.news_embeds(news_ids_tensor).unsqueeze(0)  # (1, num_news, dim)

            # Use a dummy query (e.g., average) to attend over the context
            query = context_news_embeds.mean(dim=1, keepdim=True)

            # Get context-aware word embedding
            attn_output, _ = self.word_attention(query, context_news_embeds, context_news_embeds)
            all_word_embeds[word_idx] = attn_output.squeeze(0)

        # --- Coarse-grained Step ---
        # Pad news_word_map to max length
        padded_news_word_map = nn.utils.rnn.pad_sequence(
            [torch.LongTensor(seq) for seq in news_word_map],
            batch_first=True,
            padding_value=0
        )[:, :self.max_sent_len]

        # Get embeddings for all words in all news
        doc_word_embeds = F.embedding(padded_news_word_map.to(self.device), all_word_embeds)

        x = doc_word_embeds.permute(0, 2, 1)  # (batch, dim, len)

        # Apply CNNs
        conv_outputs = [F.relu(conv(x)) for conv in self.cnn]
        pooled_outputs = [F.max_pool1d(out, out.shape[2]).squeeze(2) for out in conv_outputs]

        x = torch.cat(pooled_outputs, dim=1)
        x = self.dropout(x)
        final_news_semantic_embeds = self.fc(x)

        return final_news_semantic_embeds


# Module 2b: Meta-path Interaction Learning
class MetaPathInteractionModule(nn.Module):
    def __init__(self, num_nodes_dict, embed_dim, config):
        super().__init__()
        self.config = config
        self.num_news = num_nodes_dict['news']

        # Node embeddings
        self.news_embed = nn.Embedding(num_nodes_dict['news'], embed_dim)
        self.user_embed = nn.Embedding(num_nodes_dict['users'], embed_dim)
        self.source_embed = nn.Embedding(num_nodes_dict['sources'], embed_dim)

        # Meta-path instance learning (Multi-head Self-Attention)
        self.mhsa_nsn = nn.MultiheadAttention(embed_dim, config.ATTN_HEADS, batch_first=True)
        self.mhsa_nun = nn.MultiheadAttention(embed_dim, config.ATTN_HEADS, batch_first=True)

    def _sample_and_embed_instances(self, start_node, adj1, adj2, embed1, embed2):
        instances = []
        for _ in range(self.config.META_PATHS_SAMPLES):
            # Path: start_node -> neighbor1 -> neighbor2
            if not adj1.get(start_node): continue
            neighbor1 = random.choice(adj1[start_node])

            if not adj2.get(neighbor1) or len(adj2[neighbor1]) < 2: continue
            path_end_node = random.choice([n for n in adj2[neighbor1] if n != start_node])

            # Encode instance by averaging node embeddings
            start_embed = self.news_embed(torch.LongTensor([start_node]).to(self.config.DEVICE))
            n1_embed = embed1(torch.LongTensor([neighbor1]).to(self.config.DEVICE))
            end_embed = embed2(torch.LongTensor([path_end_node]).to(self.config.DEVICE))
            instance_embed = torch.mean(torch.cat([start_embed, n1_embed, end_embed], dim=0), dim=0)
            instances.append(instance_embed)

        if not instances:
            return torch.zeros(1, self.config.EMBED_DIM).to(self.config.DEVICE)
        return torch.stack(instances)

    def forward(self, news_ids, data_handler):
        # This is also heavy, done on the fly for clarity
        all_news_nsn_embeds = []
        all_news_nun_embeds = []

        for n_id in news_ids:
            # N-S-N path instances
            nsn_instances = self._sample_and_embed_instances(n_id, data_handler.news_source_adj,
                                                             data_handler.source_news_adj, self.source_embed,
                                                             self.news_embed).unsqueeze(0)
            # N-U-N path instances
            nun_instances = self._sample_and_embed_instances(n_id, data_handler.news_user_adj,
                                                             data_handler.user_news_adj, self.user_embed,
                                                             self.news_embed).unsqueeze(0)

            # Apply self-attention over instances
            nsn_attn, _ = self.mhsa_nsn(nsn_instances, nsn_instances, nsn_instances)
            nun_attn, _ = self.mhsa_nun(nun_instances, nun_instances, nun_instances)

            all_news_nsn_embeds.append(nsn_attn.mean(dim=1))
            all_news_nun_embeds.append(nun_attn.mean(dim=1))

        return torch.cat(all_news_nsn_embeds), torch.cat(all_news_nun_embeds)


# Module 2c: Aggregation and Final Model
class MGMP(nn.Module):
    def __init__(self, data_handler, config):
        super().__init__()
        self.config = config
        self.data_handler = data_handler

        # Semantic Module
        self.semantic_module = MultiGranularitySemanticModule(
            num_news=data_handler.num_news,
            vocab_size=data_handler.vocab_size,
            embed_dim=config.EMBED_DIM,
            cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
            cnn_out_channels=config.CNN_OUT_CHANNELS,
            attn_heads=config.ATTN_HEADS,
            dropout=config.DROPOUT,
            max_sent_len=config.MAX_SENT_LEN,
            device=config.DEVICE
        )

        # Structural Module
        num_nodes_dict = {'news': data_handler.num_news, 'users': data_handler.num_users,
                          'sources': data_handler.num_sources}
        self.structural_module = MetaPathInteractionModule(num_nodes_dict, config.EMBED_DIM, config)

        # Meta-path Aggregation (simple attention)
        self.meta_path_attention = nn.Sequential(
            nn.Linear(config.EMBED_DIM, 1),
            nn.Tanh()
        )

        # Final Classifier
        # Input: semantic_embed + aggregated_structural_embed
        self.classifier = nn.Sequential(
            nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.EMBED_DIM, 2)
        )

    def forward(self, news_ids):
        # 1. Get Semantic Embeddings
        # In a real high-performance setup, these would be pre-computed per epoch
        semantic_embeds = self.semantic_module(
            news_ids, self.data_handler.word_news_map, self.data_handler.news_word_map
        )
        # Select embeddings for the current batch
        batch_semantic_embeds = semantic_embeds[news_ids]

        # 2. Get Structural Embeddings
        nsn_embeds, nun_embeds = self.structural_module(news_ids, self.data_handler)

        # 3. Aggregate Meta-path Embeddings
        # Stack for attention: (batch, num_meta_paths, dim)
        meta_path_stack = torch.stack([nsn_embeds, nun_embeds], dim=1)

        # Compute attention weights
        attn_weights = self.meta_path_attention(meta_path_stack)  # (batch, num_meta_paths, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        aggregated_structural_embeds = (meta_path_stack * attn_weights).sum(dim=1)

        # 4. Final Classification
        combined_embeds = torch.cat([batch_semantic_embeds, aggregated_structural_embeds], dim=1)
        logits = self.classifier(combined_embeds)

        return logits


# --- 3. Training and Evaluation Loop ---

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for news_ids, labels in tqdm(data_loader, desc="Training"):
        news_ids, labels = news_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(news_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for news_ids, labels in tqdm(data_loader, desc="Evaluating"):
            news_ids, labels = news_ids.to(device), labels.to(device)
            logits = model(news_ids)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return acc, precision, recall, f1


class NewsDataset(Dataset):
    def __init__(self, df, indices):
        self.labels = df.iloc[indices]['label'].values
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx]


def main():
    config = Config()

    # 1. Load data
    data_handler = MGMPDataset(config)
    train_idx, val_idx, test_idx = data_handler.get_train_val_test_split()

    train_dataset = NewsDataset(data_handler.df, train_idx)
    val_dataset = NewsDataset(data_handler.df, val_idx)
    test_dataset = NewsDataset(data_handler.df, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # 2. Initialize model
    model = MGMP(data_handler, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # 3. Training loop
    print("\n--- Starting Training ---")
    for epoch in range(config.EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, config.DEVICE)
        val_acc, val_f1, val_auc = evaluate(model, val_loader, config.DEVICE)
        print(
            f"Epoch {epoch + 1:02d}/{config.EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

    # 4. Final evaluation
    print("\n--- Final Evaluation on Test Set ---")
    test_acc, test_pre, test_rec, test_f1 = evaluate(model, test_loader, config.DEVICE)
    # 更新打印语句
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_pre:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")


if __name__ == '__main__':
    main()