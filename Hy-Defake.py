import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")


# --- 1. 数据集与超图构建 ---
class SpamHypergraphDataset(Dataset):
    def __init__(self, data_path, tokenizer, test_size=0.2, val_size=0.1):
        """
        初始化数据集，加载数据并构建超图。
        :param data_path: CSV文件路径。
        :param tokenizer: 预训练语言模型的tokenizer。
        :param test_size: 测试集比例。
        :param val_size: 验证集比例。
        """
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 加载和预处理数据
        df = self._load_data(data_path)

        # 构建映射
        self._build_mappings(df)

        # 构建超图
        self.user_features = self._build_user_features(df)
        self.biz_contents, self.biz_labels = self._build_biz_features_and_labels(df)
        self.incidence_matrix = self._build_incidence_matrix(df)

        # 划分数据集
        self._split_dataset(test_size, val_size)

    def _load_data(self, data_path):
        """加载CSV并进行基本预处理"""
        print("Loading data...")
        df = pd.read_csv(data_path)
        # 将 'is_recommended' 列转换为标签 (False -> 1 (虚假), True -> 0 (真实))
        df['label'] = (~df['is_recommended']).astype(int)
        # 填充用户特征中的NaN值
        user_feature_cols = ['author_friend_sum', 'author_review_sum', 'author_photo_sum']
        for col in user_feature_cols:
            df[col] = df[col].fillna(0)
        df['content'] = df['content'].fillna('').astype(str)
        return df

    def _build_mappings(self, df):
        """为用户和商家创建从ID到索引的映射"""
        print("Building mappings...")
        self.unique_users = df['author_id'].unique()
        self.unique_biz = df['biz_alias'].unique()

        self.user_to_idx = {uid: i for i, uid in enumerate(self.unique_users)}
        self.biz_to_idx = {bid: i for i, bid in enumerate(self.unique_biz)}

        self.num_users = len(self.unique_users)
        self.num_biz = len(self.unique_biz)
        print(f"Found {self.num_users} unique users (nodes).")
        print(f"Found {self.num_biz} unique businesses (hyperedges).")

    def _build_user_features(self, df):
        """构建并标准化用户特征矩阵"""
        print("Building user features...")
        # 去重，确保每个用户只有一行特征
        user_df = df.drop_duplicates(subset=['author_id']).set_index('author_id')
        user_features = np.zeros((self.num_users, 3))

        for user_id, idx in self.user_to_idx.items():
            try:
                features = user_df.loc[user_id][['author_friend_sum', 'author_review_sum', 'author_photo_sum']].values
                user_features[idx] = features
            except KeyError:
                continue  # 如果用户ID不在索引中，则特征为0

        # 标准化特征
        scaler = StandardScaler()
        user_features = scaler.fit_transform(user_features)
        return torch.FloatTensor(user_features).to(self.device)

    def _build_biz_features_and_labels(self, df):
        """构建商家的文本内容和标签"""
        print("Building business (hyperedge) features and labels...")
        biz_contents = [""] * self.num_biz
        biz_labels = np.zeros(self.num_biz, dtype=int)

        # 按商家分组，聚合评论内容
        for biz_alias, group in tqdm(df.groupby('biz_alias'), desc="Aggregating biz content"):
            if biz_alias in self.biz_to_idx:
                idx = self.biz_to_idx[biz_alias]
                # 将所有评论拼接，用特殊符号分隔
                content = " [SEP] ".join(group['content'].tolist())
                biz_contents[idx] = content
                # 使用该商家的第一个评论的标签作为商家的标签
                biz_labels[idx] = group['label'].iloc[0]

        return biz_contents, torch.LongTensor(biz_labels).to(self.device)

    def _build_incidence_matrix(self, df):
        """构建超图的关联矩阵 H (用户-商家)"""
        print("Building incidence matrix...")
        H = torch.zeros((self.num_users, self.num_biz))
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Populating incidence matrix"):
            user_idx = self.user_to_idx.get(row['author_id'])
            biz_idx = self.biz_to_idx.get(row['biz_alias'])
            if user_idx is not None and biz_idx is not None:
                H[user_idx, biz_idx] = 1

        # 转换为稀疏矩阵以节省内存和加速计算
        return H.to_sparse().to(self.device)

    def _split_dataset(self, test_size, val_size):
        """划分商家索引为训练、验证和测试集"""
        indices = np.arange(self.num_biz)
        labels = self.biz_labels.cpu().numpy()

        # 第一次划分：训练+验证集 vs 测试集
        train_val_indices, self.test_indices = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=42
        )

        # 第二次划分：训练集 vs 验证集
        train_val_labels = labels[train_val_indices]
        # 计算新的验证集比例
        relative_val_size = val_size / (1.0 - test_size)
        self.train_indices, self.val_indices = train_test_split(
            train_val_indices, test_size=relative_val_size, stratify=train_val_labels, random_state=42
        )
        print(
            f"Train samples: {len(self.train_indices)}, Val samples: {len(self.val_indices)}, Test samples: {len(self.test_indices)}")

    def get_data(self):
        """返回所有构建好的数据"""
        return {
            'user_features': self.user_features,
            'biz_contents': self.biz_contents,
            'biz_labels': self.biz_labels,
            'incidence_matrix': self.incidence_matrix,
            'train_mask': self._indices_to_mask(self.train_indices),
            'val_mask': self._indices_to_mask(self.val_indices),
            'test_mask': self._indices_to_mask(self.test_indices)
        }

    def _indices_to_mask(self, indices):
        mask = torch.zeros(self.num_biz, dtype=torch.bool)
        mask[indices] = True
        return mask.to(self.device)


# --- 2. 模型模块定义 ---

class NewsSemanticChannel(nn.Module):
    """新闻语义通道，使用RoBERTa提取文本特征"""

    def __init__(self, model_name='roberta-base', device='cpu'):  # <--- 修改默认模型名称
        super().__init__()
        # 从 DistilBertTokenizer 更改为 RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        # 从 DistilBertModel 更改为 RobertaModel
        self.bert = RobertaModel.from_pretrained(model_name)
        self.device = device

    def forward(self, text_list, batch_size=16):  # <--- 建议减小batch_size，因为RoBERTa更耗显存
        all_embeddings = []
        self.bert.eval()  # 冻结模型
        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc="News Semantic Channel (RoBERTa)"):
                batch_texts = text_list[i:i + batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                        max_length=512).to(self.device)
                outputs = self.bert(**inputs)
                # 使用[CLS] token的输出来代表整个文本序列
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).to(self.device)


class HGNNConv(nn.Module):
    """超图卷积层"""

    def __init__(self, in_dim, out_dim, use_bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X, H):
        # X: 节点特征 (num_users, in_dim)
        # H: 关联矩阵 (num_users, num_biz)
        X = X @ self.weight
        if self.bias is not None:
            X = X + self.bias

        # 节点度
        D_v = torch.sparse.sum(H, dim=1).to_dense().pow(-0.5)
        D_v[torch.isinf(D_v)] = 0
        D_v = torch.diag(D_v)

        # 超边度
        D_e = torch.sparse.sum(H, dim=0).to_dense().pow(-1)
        D_e[torch.isinf(D_e)] = 0
        D_e = torch.diag(D_e)

        # H_T * D_v * X
        step1 = torch.sparse.mm(H.t(), D_v @ X)
        # D_e * H_T * D_v * X
        step2 = D_e @ step1
        # H * D_e * H_T * D_v * X
        step3 = torch.sparse.mm(H, step2)
        # D_v * H * D_e * H_T * D_v * X
        final_X = D_v @ step3

        return final_X


class UserCredibilityChannel(nn.Module):
    """用户信誉通道 (超图自动编码器)"""

    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super().__init__()
        # Encoder
        self.hgnn1 = HGNNConv(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Decoder (简单的线性解码器)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def forward(self, X, H):
        # 编码
        encoded_features = self.hgnn1(X, H)
        encoded_features = F.relu(encoded_features)
        encoded_features = self.dropout(encoded_features)

        # 解码
        reconstructed_features = self.decoder(encoded_features)

        return encoded_features, reconstructed_features


class HyDeFake(nn.Module):
    """Hy-DeFake 主模型"""

    def __init__(self, user_in_dim, user_hidden_dim, news_embed_dim, num_classes=2, dropout=0.5):
        super().__init__()
        self.user_credibility_channel = UserCredibilityChannel(user_in_dim, user_hidden_dim, dropout)

        # 融合后的特征维度
        fused_dim = news_embed_dim + user_hidden_dim

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes)
        )

    def forward(self, user_features, news_features, incidence_matrix):
        # 1. 用户信誉通道
        user_embeds, user_reconstructed = self.user_credibility_channel(user_features, incidence_matrix)

        # 2. 特征融合
        # 计算每个超边的平均用户嵌入
        # D_e_inv: (num_biz, num_biz)
        D_e_inv = torch.sparse.sum(incidence_matrix, dim=0).to_dense().pow(-1)
        D_e_inv[torch.isinf(D_e_inv)] = 0
        D_e_inv = torch.diag(D_e_inv)

        # H.t() @ user_embeds -> (num_biz, user_hidden_dim), 是每个超边上用户特征的总和
        # D_e_inv @ (H.t() @ user_embeds) -> 平均用户特征
        aggregated_user_embeds = D_e_inv @ torch.sparse.mm(incidence_matrix.t(), user_embeds)

        # 拼接新闻语义特征和聚合后的用户特征
        fused_features = torch.cat([news_features, aggregated_user_embeds], dim=1)

        # 3. 分类
        logits = self.classifier(fused_features)

        return logits, user_reconstructed


# --- 3. 训练与评估 ---

def evaluate(model, data, mask):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        logits, _ = model(data['user_features'], data['news_features'], data['incidence_matrix'])
        logits = logits[mask]
        labels = data['biz_labels'][mask]

        preds = torch.argmax(logits, dim=1)

        # 转换为cpu上的numpy数组进行评估
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        acc = accuracy_score(labels_np, preds_np)
        pre = precision_score(labels_np, preds_np, zero_division=0)
        rec = recall_score(labels_np, preds_np, zero_division=0)
        f1 = f1_score(labels_np, preds_np, zero_division=0)

        return acc, pre, rec, f1


def main():
    # --- 参数设置 ---
    DATA_PATH = '../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv'
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    USER_HIDDEN_DIM = 128
    NEWS_EMBED_DIM = 768  # roberta-base 的输出维度也是 768
    DROPOUT = 0.5
    RECONSTRUCTION_WEIGHT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据加载 ---
    # 将 tokenizer 初始化更改为 RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # <--- 修改此处
    dataset = SpamHypergraphDataset(DATA_PATH, tokenizer)
    data = dataset.get_data()

    # --- 预计算新闻特征 ---
    print("Pre-computing news features with RoBERTa...")
    # 此处会自动使用 roberta-base 模型
    news_channel = NewsSemanticChannel(model_name='roberta-base', device="cuda")  # <--- 明确指定模型
    news_channel.to(device)
    data['news_features'] = news_channel(data['biz_contents'])
    print("News features computed.")

    # --- 模型初始化 ---
    model = HyDeFake(
        user_in_dim=data['user_features'].shape[1],
        user_hidden_dim=USER_HIDDEN_DIM,
        news_embed_dim=NEWS_EMBED_DIM,
        num_classes=2,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    classification_loss_fn = nn.CrossEntropyLoss()
    reconstruction_loss_fn = nn.MSELoss()

    # --- 训练循环 ---
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        logits, user_reconstructed = model(data['user_features'], data['news_features'], data['incidence_matrix'])

        # 计算损失
        cls_loss = classification_loss_fn(logits[data['train_mask']], data['biz_labels'][data['train_mask']])
        rec_loss = reconstruction_loss_fn(user_reconstructed, data['user_features'])

        total_loss = cls_loss + RECONSTRUCTION_WEIGHT * rec_loss

        total_loss.backward()
        optimizer.step()

        # 在验证集上评估
        val_acc, val_pre, val_rec, val_f1 = evaluate(model, data, data['val_mask'])

        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss: {total_loss.item():.4f} "
              f"(Cls: {cls_loss.item():.4f}, Rec: {rec_loss.item():.4f}) | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # --- 最终测试 ---
    print("\n--- Final Evaluation on Test Set ---")
    test_acc, test_pre, test_rec, test_f1 = evaluate(model, data, data['test_mask'])
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_pre:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")


if __name__ == '__main__':
    main()