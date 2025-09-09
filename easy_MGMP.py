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
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5


print(f"Using device: {Config.DEVICE}")
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


class NewsDataset(Dataset):
    def __init__(self, df, indices, news_word_map, max_len):
        self.labels = df.iloc[indices]['label'].values
        self.indices = indices

        # 预处理新闻的词索引序列
        self.news_sequences = []
        for idx in indices:
            seq = news_word_map[idx]
            # 填充和截断
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [0] * (max_len - len(seq))
            self.news_sequences.append(torch.LongTensor(seq))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx], self.news_sequences[idx]

# 它现在只是一个简单的CNN编码器，不再处理词级别注意力

class CoarseGrainedEncoder(nn.Module):
    def __init__(self, embed_dim, config):
        super().__init__()
        self.config = config
        self.cnn = nn.ModuleList([
            nn.Conv1d(embed_dim, config.CNN_OUT_CHANNELS, k) for k in config.CNN_KERNEL_SIZES
        ])
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(len(config.CNN_KERNEL_SIZES) * config.CNN_OUT_CHANNELS, embed_dim)

    def forward(self, doc_word_embeds):
        # 输入: (batch_size, max_len, dim)
        x = doc_word_embeds.permute(0, 2, 1)  # (batch, dim, len)

        conv_outputs = [F.relu(conv(x)) for conv in self.cnn]
        pooled_outputs = [F.max_pool1d(out, out.shape[2]).squeeze(2) for out in conv_outputs]

        x = torch.cat(pooled_outputs, dim=1)
        x = self.dropout(x)
        final_news_semantic_embeds = self.fc(x)

        return final_news_semantic_embeds


# 新增一个函数，用于在一个 epoch 开始前计算所有内容的特征
def precompute_epoch_features(data_handler, initial_news_embeds, word_attention_model, device, config):
    """在一个 epoch 开始前，预计算细粒度的词向量和粗粒度的文档词序列"""
    print("Pre-computing features for the epoch...")

    # --- 1. 细粒度词向量计算 ---
    word_attention_model.eval()
    with torch.no_grad():
        all_word_embeds = torch.zeros(data_handler.vocab_size, config.EMBED_DIM).to(device)

        for word_idx, news_indices in tqdm(data_handler.word_news_map.items(), desc="Fine-grained word learning"):
            if not news_indices: continue
            news_ids_tensor = torch.LongTensor(news_indices).to(device)
            # 使用初始的新聞 embedding
            context_news_embeds = F.embedding(news_ids_tensor, initial_news_embeds).unsqueeze(0)

            query = context_news_embeds.mean(dim=1, keepdim=True)
            attn_output, _ = word_attention_model(query, context_news_embeds, context_news_embeds)
            all_word_embeds[word_idx] = attn_output.squeeze(0)

    # --- 2. 粗粒度文档词序列构建 ---
    padded_news_word_map = nn.utils.rnn.pad_sequence(
        [torch.LongTensor(seq) for seq in data_handler.news_word_map],
        batch_first=True,
        padding_value=0
    )[:, :config.MAX_SENT_LEN].to(device)

    # 将词索引序列转换为词向量序列
    doc_word_embeds = F.embedding(padded_news_word_map, all_word_embeds)

    return doc_word_embeds


# --- 修改后的结构模块 ---

# --- 升级后的结构模块 ---
class MetaPathInteractionModule(nn.Module):
    def __init__(self, embed_dim, config):
        super().__init__()
        self.config = config
        # GRU来编码一个元路径实例 (路径长度为3)
        self.path_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True
        )

        # 多头自注意力机制，用于聚合一个新闻的所有元路径实例
        self.mhsa = nn.MultiheadAttention(embed_dim, config.ATTN_HEADS, batch_first=True, dropout=config.DROPOUT)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, path_instances_embeds):
        # 输入 path_instances_embeds: (batch_size, num_samples, path_len, dim)
        batch_size, num_samples, path_len, dim = path_instances_embeds.shape

        # 展平以适配GRU
        path_instances_flat = path_instances_embeds.view(batch_size * num_samples, path_len, dim)

        # GRU编码，我们取最后一个时间步的 hidden state 作为整个路径的编码
        _, hidden_state = self.path_encoder(path_instances_flat)
        # hidden_state: (num_layers, batch*num_samples, dim) -> (batch*num_samples, dim)
        encoded_paths = hidden_state.squeeze(0)

        # 恢复形状: (batch_size, num_samples, dim)
        encoded_paths = encoded_paths.view(batch_size, num_samples, dim)

        # 自注意力聚合
        attn_output, _ = self.mhsa(encoded_paths, encoded_paths, encoded_paths)

        # 残差连接和归一化
        aggregated_embeds = self.norm(encoded_paths + attn_output)

        # 返回每个新闻聚合后的元路径表示 (取平均)
        return aggregated_embeds.mean(dim=1)


# 新增一个函数，用于预采样元路径
def pre_sample_meta_paths(data_handler, news_embeds, user_embeds, source_embeds, device, config):
    print("Pre-sampling all meta-path instances...")
    all_nsn_instances = []
    all_nun_instances = []

    news_embeds, user_embeds, source_embeds = news_embeds.weight.data, user_embeds.weight.data, source_embeds.weight.data

    for n_id in tqdm(range(data_handler.num_news), desc="Sampling Meta-paths"):
        # --- N-S-N Sampling ---
        nsn_samples = []
        for _ in range(config.META_PATHS_SAMPLES):
            if not data_handler.news_source_adj.get(n_id): continue
            s_idx = random.choice(data_handler.news_source_adj[n_id])
            if not data_handler.source_news_adj.get(s_idx) or len(data_handler.source_news_adj[s_idx]) < 2: continue
            end_n_id = random.choice([n for n in data_handler.source_news_adj[s_idx] if n != n_id])

            instance_embed = torch.mean(torch.stack([news_embeds[n_id], source_embeds[s_idx], news_embeds[end_n_id]]),
                                        dim=0)
            nsn_samples.append(instance_embed)

        if not nsn_samples: nsn_samples.append(torch.zeros(config.EMBED_DIM).to(device))
        all_nsn_instances.append(torch.stack(nsn_samples))

        # --- N-U-N Sampling ---
        nun_samples = []
        for _ in range(config.META_PATHS_SAMPLES):
            if not data_handler.news_user_adj.get(n_id): continue
            u_idx = random.choice(data_handler.news_user_adj[n_id])
            if not data_handler.user_news_adj.get(u_idx) or len(data_handler.user_news_adj[u_idx]) < 2: continue
            end_n_id = random.choice([n for n in data_handler.user_news_adj[u_idx] if n != n_id])

            instance_embed = torch.mean(torch.stack([news_embeds[n_id], user_embeds[u_idx], news_embeds[end_n_id]]),
                                        dim=0)
            nun_samples.append(instance_embed)

        if not nun_samples: nun_samples.append(torch.zeros(config.EMBED_DIM).to(device))
        all_nun_instances.append(torch.stack(nun_samples))

    # 填充长度不一的采样结果
    nsn_padded = nn.utils.rnn.pad_sequence(all_nsn_instances, batch_first=True, padding_value=0)
    nun_padded = nn.utils.rnn.pad_sequence(all_nun_instances, batch_first=True, padding_value=0)

    return nsn_padded, nun_padded


# --- 修改后的主模型 ---

# --- 最终版主模型 ---
class MGMP(nn.Module):
    def __init__(self, data_handler, config):
        super().__init__()
        self.config = config
        self.data_handler = data_handler

        # --- 1. 所有可学习的嵌入层 ---
        self.news_embeds = nn.Embedding(data_handler.num_news, config.EMBED_DIM)
        self.user_embeds = nn.Embedding(data_handler.num_users, config.EMBED_DIM)
        self.source_embeds = nn.Embedding(data_handler.num_sources, config.EMBED_DIM)
        self.word_embeds = nn.Embedding(data_handler.vocab_size, config.EMBED_DIM, padding_idx=0)

        # --- 2. 语义学习模块 ---
        # 细粒度词向量学习的注意力模块
        self.word_attention = nn.MultiheadAttention(config.EMBED_DIM, config.ATTN_HEADS, batch_first=True)
        # 粗粒度文档学习的CNN编码器
        self.coarse_encoder = CoarseGrainedEncoder(config.EMBED_DIM, config)

        # --- 3. 结构学习模块 ---
        self.interaction_nsn = MetaPathInteractionModule(config.EMBED_DIM, config)
        self.interaction_nun = MetaPathInteractionModule(config.EMBED_DIM, config)

        # --- 4. 聚合与分类 ---
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
        """为一批词计算其细粒度表示"""
        word_embeds_list = []
        for word_idx in word_indices:
            news_context_ids = self.data_handler.word_news_map.get(word_idx.item())
            if not news_context_ids:
                # 如果词没有上下文（不太可能，除非是罕见词），用其自身的原始嵌入
                word_embeds_list.append(self.word_embeds(word_idx))
                continue

            news_context_ids = torch.LongTensor(news_context_ids[:50]).to(self.config.DEVICE)  # 限制上下文数量以防OOM
            context_news_embeds = self.news_embeds(news_context_ids).unsqueeze(0)

            query = self.word_embeds(word_idx).unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
            attn_output, _ = self.word_attention(query, context_news_embeds, context_news_embeds)
            word_embeds_list.append(attn_output.squeeze(0).squeeze(0))

        return torch.stack(word_embeds_list)

    def _sample_meta_paths_for_batch(self, news_ids, path_type):
        """为当前批次高效采样元路径并获取嵌入"""
        batch_instances = []
        for n_id in news_ids:
            node_instances = []
            if path_type == 'NSN':
                adj1, adj2, embed1, embed2 = self.data_handler.news_source_adj, self.data_handler.source_news_adj, self.source_embeds, self.news_embeds
            else:  # NUN
                adj1, adj2, embed1, embed2 = self.data_handler.news_user_adj, self.data_handler.user_news_adj, self.user_embeds, self.news_embeds

            for _ in range(self.config.META_PATHS_SAMPLES):
                if not adj1.get(n_id.item()): continue
                neighbor1_idx = random.choice(adj1[n_id.item()])
                if not adj2.get(neighbor1_idx) or len(adj2[neighbor1_idx]) < 2: continue
                end_node_idx = random.choice([n for n in adj2[neighbor1_idx] if n != n_id.item()])

                # 获取路径中节点的嵌入
                start_embed = self.news_embeds(n_id)
                n1_embed = embed1(torch.LongTensor([neighbor1_idx]).to(self.config.DEVICE)).squeeze(0)
                end_embed = embed2(torch.LongTensor([end_node_idx]).to(self.config.DEVICE)).squeeze(0)
                node_instances.append(torch.stack([start_embed, n1_embed, end_embed]))

            if not node_instances:
                node_instances.append(torch.zeros(3, self.config.EMBED_DIM).to(self.config.DEVICE))

            # 填充以确保每个新闻的采样数量一致
            while len(node_instances) < self.config.META_PATHS_SAMPLES:
                node_instances.append(torch.zeros(3, self.config.EMBED_DIM).to(self.config.DEVICE))

            batch_instances.append(torch.stack(node_instances))

        return torch.stack(batch_instances)

    def forward(self, news_ids, news_word_indices):
        # --- 1. 语义学习 ---
        # 获取当前批次新闻中所有不重复的词
        unique_word_indices = torch.unique(news_word_indices.flatten())
        # 计算这些词的细粒度嵌入
        fine_grained_embeds = self._get_fine_grained_word_embeds(unique_word_indices)

        # 创建一个临时的词嵌入表，用于当前批次
        temp_word_embed_table = self.word_embeds.weight.clone()
        temp_word_embed_table[unique_word_indices] = fine_grained_embeds

        # 使用更新后的词嵌入表来获取文档的词序列嵌入
        doc_word_embeds = F.embedding(news_word_indices, temp_word_embed_table, padding_idx=0)

        # 通过CNN得到最终语义表示
        semantic_embeds = self.coarse_encoder(doc_word_embeds)

        # --- 2. 结构学习 ---
        nsn_instances = self._sample_meta_paths_for_batch(news_ids, 'NSN')
        nun_instances = self._sample_meta_paths_for_batch(news_ids, 'NUN')

        nsn_embeds = self.interaction_nsn(nsn_instances)
        nun_embeds = self.interaction_nun(nun_instances)

        # --- 3. 聚合与分类 ---
        meta_path_stack = torch.stack([nsn_embeds, nun_embeds], dim=1)
        attn_weights = F.softmax(self.meta_path_attention(meta_path_stack), dim=1)
        aggregated_structural_embeds = (meta_path_stack * attn_weights).sum(dim=1)

        combined_embeds = torch.cat([semantic_embeds, aggregated_structural_embeds], dim=1)
        logits = self.classifier(combined_embeds)

        return logits


# --- 修改后的训练和主函数 ---

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for news_ids, labels, news_sequences in tqdm(data_loader, desc="Training"):
        news_ids, labels, news_sequences = news_ids.to(device), labels.to(device), news_sequences.to(device)

        optimizer.zero_grad()
        logits = model(news_ids, news_sequences)
        loss = criterion(logits, labels)
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
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

    # 将 news_word_map 和 max_len 传入 Dataset
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
        # --- 在每个 epoch 开始时进行预计算 ---
        # 1. 预计算语义特征
        precomputed_doc_embeds = precompute_epoch_features(
            data_handler, model.initial_news_embeds.weight, model.word_attention, config.DEVICE, config
        )
        # 2. 预采样元路径
        precomputed_nsn, precomputed_nun = pre_sample_meta_paths(
            data_handler, model.initial_news_embeds, model.user_embeds, model.source_embeds, config.DEVICE, config
        )

        # 传递预计算的特征进行训练
        train_loss = train(model, train_loader, optimizer, criterion, precomputed_doc_embeds, precomputed_nsn,
                           precomputed_nun, config.DEVICE)

        # 传递预计算的特征进行评估
        val_acc, val_pre, val_rec, val_f1 = evaluate(model, val_loader, precomputed_doc_embeds, precomputed_nsn,
                                                     precomputed_nun, config.DEVICE)

        print(f"Epoch {epoch + 1:02d}/{config.EPOCHS} | Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Pre: {val_pre:.4f} | "
              f"Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")

    print("\n--- Final Evaluation on Test Set ---")
    # 最终测试也需要预计算
    final_doc_embeds = precompute_epoch_features(data_handler, model.initial_news_embeds.weight, model.word_attention,
                                                 config.DEVICE, config)
    final_nsn, final_nun = pre_sample_meta_paths(data_handler, model.initial_news_embeds, model.user_embeds,
                                                 model.source_embeds, config.DEVICE, config)

    test_acc, test_pre, test_rec, test_f1 = evaluate(model, test_loader, final_doc_embeds, final_nsn, final_nun,
                                                     config.DEVICE)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_pre:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")


if __name__ == '__main__':
    main()
