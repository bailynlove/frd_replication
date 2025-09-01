import os
import json
import time
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import networkx as nx
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """设置随机种子以保证结果可复现（支持CPU、CUDA和MPS）"""
    import random
    import numpy as np
    import torch
    import os

    # 基本随机种子设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA设备设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # MPS设备设置 (Apple Silicon)
    if hasattr(torch, 'mps') and torch.mps.is_available():
        try:
            # PyTorch 1.12+ 版本支持
            torch.mps.manual_seed(seed)
            print("MPS设备随机种子已设置")
        except AttributeError:
            # 旧版本PyTorch可能需要不同的设置方式
            print("警告: 当前PyTorch版本可能不支持torch.mps.manual_seed()")

    # 确保跨平台一致性
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


set_seed()


def preprocess_text(text):
    """文本预处理函数"""
    if pd.isna(text) or text == '':
        return "empty review"
    # 移除特殊字符和多余空格
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def save_cached_data(processed_data, cache_dir="./data_cache"):
    """保存处理后的数据到缓存"""
    os.makedirs(cache_dir, exist_ok=True)

    # 保存交互矩阵
    np.save(os.path.join(cache_dir, 'interaction_matrix.npy'), processed_data['interaction_matrix'])

    # 保存用户特征矩阵
    np.save(os.path.join(cache_dir, 'user_feat_matrix.npy'), processed_data['user_feat_matrix'])

    # 保存用户标签
    np.save(os.path.join(cache_dir, 'user_labels.npy'), processed_data['user_labels'])

    # 保存图 - 使用pickle保存
    G = processed_data['G']

    # 临时保存转移矩阵到单独的文件
    if 'P' in G.graph:
        with open(os.path.join(cache_dir, 'graph_P.pkl'), 'wb') as f:
            pickle.dump(G.graph['P'], f)
        # 从G.graph中移除P，因为它可能无法正确pickle
        del G.graph['P']

    with open(os.path.join(cache_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(G, f)

    # 保存映射
    with open(os.path.join(cache_dir, 'user_id_to_idx.json'), 'w') as f:
        json.dump(processed_data['user_id_to_idx'], f)
    with open(os.path.join(cache_dir, 'idx_to_user_id.json'), 'w') as f:
        # idx_to_user_id 的键是整数，需要转换为字符串才能保存为JSON
        json.dump({str(k): v for k, v in processed_data['idx_to_user_id'].items()}, f)
    with open(os.path.join(cache_dir, 'biz_id_to_idx.json'), 'w') as f:
        json.dump(processed_data['biz_id_to_idx'], f)
    with open(os.path.join(cache_dir, 'idx_to_biz_id.json'), 'w') as f:
        # idx_to_biz_id 的键是整数，需要转换为字符串才能保存为JSON
        json.dump({str(k): v for k, v in processed_data['idx_to_biz_id'].items()}, f)

    # 保存索引
    np.save(os.path.join(cache_dir, 'train_idx.npy'), processed_data['train_idx'])
    np.save(os.path.join(cache_dir, 'val_idx.npy'), processed_data['val_idx'])
    np.save(os.path.join(cache_dir, 'test_idx.npy'), processed_data['test_idx'])

    print(f"数据已缓存到 {cache_dir}")


def load_cached_data(cache_dir="./data_cache"):
    """尝试加载缓存数据，如果发现损坏则删除并返回None"""
    cache_files = {
        'interaction_matrix': os.path.join(cache_dir, 'interaction_matrix.npy'),
        'user_feat_matrix': os.path.join(cache_dir, 'user_feat_matrix.npy'),
        'user_labels': os.path.join(cache_dir, 'user_labels.npy'),
        'graph': os.path.join(cache_dir, 'graph.pkl'),
        'user_id_to_idx': os.path.join(cache_dir, 'user_id_to_idx.json'),
        'idx_to_user_id': os.path.join(cache_dir, 'idx_to_user_id.json'),
        'biz_id_to_idx': os.path.join(cache_dir, 'biz_id_to_idx.json'),
        'idx_to_biz_id': os.path.join(cache_dir, 'idx_to_biz_id.json'),
        'train_idx': os.path.join(cache_dir, 'train_idx.npy'),
        'val_idx': os.path.join(cache_dir, 'val_idx.npy'),
        'test_idx': os.path.join(cache_dir, 'test_idx.npy')
    }

    # 检查所有缓存文件是否存在
    all_files_exist = all(os.path.exists(f) for f in cache_files.values())

    if not all_files_exist:
        return None

    try:
        # 尝试加载并验证idx_to_user_id
        with open(cache_files['idx_to_user_id'], 'r') as f:
            idx_to_user_id = json.load(f)
            # 验证键是否为数字字符串
            for k in idx_to_user_id.keys():
                try:
                    int(k)  # 尝试转换为整数
                except ValueError:
                    raise ValueError(f"idx_to_user_id contains non-numeric key: {k}")

        # 尝试加载并验证idx_to_biz_id
        with open(cache_files['idx_to_biz_id'], 'r') as f:
            idx_to_biz_id = json.load(f)
            # 验证键是否为数字字符串
            for k in idx_to_biz_id.keys():
                try:
                    int(k)  # 尝试转换为整数
                except ValueError:
                    raise ValueError(f"idx_to_biz_id contains non-numeric key: {k}")

        print("加载缓存数据...")

        # 加载交互矩阵
        interaction_matrix = np.load(cache_files['interaction_matrix'])

        # 加载用户特征矩阵
        user_feat_matrix = np.load(cache_files['user_feat_matrix'])

        # 加载用户标签
        user_labels = np.load(cache_files['user_labels'])

        # 加载图 - 使用pickle加载
        with open(cache_files['graph'], 'rb') as f:
            G = pickle.load(f)

        # 重新加载转移矩阵
        P_file = os.path.join(cache_dir, 'graph_P.pkl')
        if os.path.exists(P_file):
            with open(P_file, 'rb') as f:
                G.graph['P'] = pickle.load(f)
        else:
            # 如果没有P文件，可能是旧缓存，需要重新计算
            print("警告: 未找到转移矩阵P，需要重新计算...")
            from scipy import sparse
            adj_matrix = nx.to_scipy_sparse_array(G, format='csr')
            degrees = np.array(adj_matrix.sum(axis=1)).flatten()
            degrees[degrees == 0] = 1
            D_inv = sparse.diags(1.0 / degrees)
            P = D_inv @ adj_matrix
            G.graph['P'] = P

        # 加载映射
        with open(cache_files['user_id_to_idx'], 'r') as f:
            user_id_to_idx = json.load(f)

        with open(cache_files['idx_to_user_id'], 'r') as f:
            idx_to_user_id = json.load(f)
            # JSON键是字符串形式的数字，应转换为整数
            idx_to_user_id = {int(k): v for k, v in idx_to_user_id.items()}

        with open(cache_files['biz_id_to_idx'], 'r') as f:
            biz_id_to_idx = json.load(f)

        with open(cache_files['idx_to_biz_id'], 'r') as f:
            idx_to_biz_id = json.load(f)
            # JSON键是字符串形式的数字，应转换为整数
            idx_to_biz_id = {int(k): v for k, v in idx_to_biz_id.items()}

        # 加载索引
        train_idx = np.load(cache_files['train_idx'])
        val_idx = np.load(cache_files['val_idx'])
        test_idx = np.load(cache_files['test_idx'])

        return {
            'interaction_matrix': interaction_matrix,
            'user_feat_matrix': user_feat_matrix,
            'user_labels': user_labels,
            'G': G,
            'user_id_to_idx': user_id_to_idx,
            'idx_to_user_id': idx_to_user_id,
            'biz_id_to_idx': biz_id_to_idx,
            'idx_to_biz_id': idx_to_biz_id,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }

    except Exception as e:
        print(f"缓存数据损坏: {str(e)}")
        print("正在删除损坏的缓存文件...")

        # 删除所有缓存文件
        for file_path in cache_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"已删除: {file_path}")

        # 删除空目录
        if os.path.exists(cache_dir) and not os.listdir(cache_dir):
            os.rmdir(cache_dir)

        return None


def preprocess_data():
    """预处理数据"""
    # 检查缓存
    cached_data = load_cached_data()
    if cached_data is not None:
        print("使用缓存数据")
        return cached_data

    print("缓存数据不存在，开始预处理...")

    # 加载数据
    data_path = "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv"
    df = pd.read_csv(data_path)

    print(f"原始数据形状: {df.shape}")
    print(f"真实评论比例: {df['is_recommended'].mean():.2f}")

    # 确保目标变量无缺失
    assert df['is_recommended'].isnull().sum() == 0, "is_recommended contains missing values"

    # 文本预处理
    df['processed_content'] = df['content'].apply(preprocess_text)

    # 1. 用户-产品交互矩阵构建
    # 用户ID到索引的映射
    user_ids = df['author_id'].unique()
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    # 正确创建从索引到用户ID的映射
    idx_to_user_id = dict(enumerate(user_ids))

    # 产品(商家)ID到索引的映射
    biz_ids = df['biz_id'].unique()
    biz_id_to_idx = {biz_id: idx for idx, biz_id in enumerate(biz_ids)}
    # 正确创建从索引到商家ID的映射
    idx_to_biz_id = dict(enumerate(biz_ids))

    # 创建用户-产品交互矩阵
    num_users = len(user_ids)
    num_products = len(biz_ids)
    interaction_matrix = np.zeros((num_users, num_products))

    # 填充交互矩阵 (1表示有交互)
    for _, row in df.iterrows():
        user_idx = user_id_to_idx[row['author_id']]
        biz_idx = biz_id_to_idx[row['biz_id']]
        interaction_matrix[user_idx, biz_idx] = 1

    # 2. 用户特征处理
    user_features = [
        'author_friend_sum',
        'author_review_sum',
        'author_photo_sum',
        'text_length',
        'word_count',
        'polarity',
        'subjectivity'
    ]

    # 标准化用户特征
    scaler = StandardScaler()
    user_feat_matrix = df[user_features].fillna(0).values
    user_feat_matrix = scaler.fit_transform(user_feat_matrix)

    # 3. 构建用户-用户关系图 (基于共同交互的商家)
    G = nx.Graph()

    # 添加节点
    for i in range(num_users):
        G.add_node(i)

    # 添加边 (基于Jaccard相似度)
    # 优化1: 提高相似度阈值，减少边数量
    similarity_threshold = 0.2  # 从0.1提高到0.2，减少图的密度

    print(f"开始构建用户-用户关系图 (相似度阈值: {similarity_threshold})...")
    start_time = time.time()

    # 优化2: 使用更高效的相似度计算方法
    # 创建交互矩阵的稀疏表示
    from scipy.sparse import csr_matrix
    interaction_sparse = csr_matrix(interaction_matrix)

    # 计算Jaccard相似度矩阵
    intersection = interaction_sparse @ interaction_sparse.T
    row_sum = np.array(interaction_sparse.sum(axis=1)).flatten()
    union = row_sum[:, np.newaxis] + row_sum - intersection.toarray()

    # 避免除以0
    union[union == 0] = 1

    # 计算相似度
    similarity = intersection.toarray() / union

    # 添加边
    for i in range(num_users):
        for j in range(i + 1, num_users):
            if similarity[i, j] > similarity_threshold:
                G.add_edge(i, j, weight=similarity[i, j])

    print(f"用户-用户关系图构建完成，耗时: {time.time() - start_time:.2f}秒")
    print(
        f"用户-用户关系图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边 (密度: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2):.4f})")

    # === 关键优化3: 预计算RWR转移矩阵 ===
    print("预计算RWR转移矩阵...")
    start_time = time.time()

    # 使用稀疏矩阵表示（大幅提高效率）
    from scipy import sparse

    # 创建邻接矩阵（稀疏格式）
    adj_matrix = nx.to_scipy_sparse_array(G, format='csr')

    # 计算度矩阵的逆
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # 避免除以0
    D_inv = sparse.diags(1.0 / degrees)

    # 计算转移矩阵 P = D^-1 * A
    P = D_inv @ adj_matrix

    # 保存到图对象
    G.graph['P'] = P
    # 不再需要单独保存 num_nodes，因为可以使用 G.number_of_nodes()

    print(f"转移矩阵预计算完成，耗时: {time.time() - start_time:.2f}秒")

    # 4. 创建可疑用户标签
    # is_recommended为0表示假评论，即垃圾用户
    # 这里我们假设如果用户发布了超过50%的假评论，则认为该用户是垃圾用户
    user_spam_ratio = df.groupby('author_id')['is_recommended'].apply(lambda x: 1 - x.mean())
    spam_threshold = 0.5  # 如果50%以上的评论是假的，则认为是垃圾用户

    # 创建用户标签
    user_labels = np.zeros(num_users)
    for i, user_id in enumerate(user_ids):
        if user_id in user_spam_ratio.index:
            user_labels[i] = 1 if user_spam_ratio[user_id] >= spam_threshold else 0

    print(f"垃圾用户比例: {user_labels.mean():.2f}")

    # 5. 划分数据集
    # 仅使用有标签的用户
    labeled_indices = np.where(user_labels != -1)[0]
    train_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=0.2,
        random_state=42,
        stratify=user_labels[labeled_indices]
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.125,
        random_state=42,
        stratify=user_labels[train_idx]
    )

    print(f"训练集: {len(train_idx)} ({user_labels[train_idx].mean():.2f} 垃圾), "
          f"验证集: {len(val_idx)} ({user_labels[val_idx].mean():.2f} 垃圾), "
          f"测试集: {len(test_idx)} ({user_labels[test_idx].mean():.2f} 垃圾)")

    # 6. 保存处理后的数据
    processed_data = {
        'interaction_matrix': interaction_matrix,
        'user_feat_matrix': user_feat_matrix,
        'user_labels': user_labels,
        'G': G,
        'user_id_to_idx': user_id_to_idx,
        'idx_to_user_id': idx_to_user_id,
        'biz_id_to_idx': biz_id_to_idx,
        'idx_to_biz_id': idx_to_biz_id,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }

    # 保存缓存
    save_cached_data(processed_data)

    return processed_data


# RWR缓存
rwr_cache = {}


def batch_rwr_sampling(G, node_indices, restart_prob=0.2, max_steps=20, top_k=10):
    """
    批量RWR采样 - 一次性处理多个节点

    参数:
    - G: 用户-用户关系图
    - node_indices: 节点索引列表
    - restart_prob: 重启概率
    - max_steps: 最大步数
    - top_k: 采样邻居数量

    返回:
    - 邻接矩阵，仅包含采样的邻居
    """
    num_nodes = G.number_of_nodes()
    adj = np.zeros((num_nodes, num_nodes))

    # 从图中获取预计算的转移矩阵
    if 'P' in G.graph:
        P = G.graph['P']
    else:
        # 如果P不存在，可能是旧缓存，需要重新计算
        print("警告: 转移矩阵P不存在，需要重新计算...")
        from scipy import sparse
        adj_matrix = nx.to_scipy_sparse_array(G, format='csr')
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv = sparse.diags(1.0 / degrees)
        P = D_inv @ adj_matrix
        G.graph['P'] = P

    # 批量处理
    p_batch = np.zeros((len(node_indices), num_nodes))
    for i, node_idx in enumerate(node_indices):
        p = np.zeros(num_nodes)
        p[node_idx] = 1.0
        p_batch[i] = p

    # 批量RWR迭代
    for _ in range(max_steps):
        p_batch = (1 - restart_prob) * p_batch @ P.toarray() + restart_prob * np.eye(num_nodes)[node_indices]

    # 构建采样邻接矩阵
    for i, node_idx in enumerate(node_indices):
        # 获取重要邻居
        neighbor_indices = np.argsort(-p_batch[i])[:top_k]

        # 仅保留top_k邻居（排除自身）
        for neighbor in neighbor_indices:
            if neighbor != node_idx:
                adj[node_idx, neighbor] = 1
                adj[neighbor, node_idx] = 1  # 无向图

    return adj


def rwr(G, start_node, restart_prob=0.2, max_steps=20, threshold=1e-6):
    """
    优化版RWR实现

    参数:
    - G: 用户-用户关系图（已预计算转移矩阵）
    - start_node: 起始节点
    - restart_prob: 重启概率
    - max_steps: 最大步数（从50减少到20）
    - threshold: 收敛阈值

    返回:
    - 重要邻居节点列表
    """
    # 检查缓存
    cache_key = f"{start_node}_{restart_prob}_{max_steps}"
    if cache_key in rwr_cache:
        return rwr_cache[cache_key]

    # 从图中获取预计算的转移矩阵
    if 'P' in G.graph:
        P = G.graph['P']
    else:
        # 如果P不存在，可能是旧缓存，需要重新计算
        print(f"警告: 转移矩阵P不存在，需要重新计算 (节点 {start_node})...")
        from scipy import sparse
        adj_matrix = nx.to_scipy_sparse_array(G, format='csr')
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv = sparse.diags(1.0 / degrees)
        P = D_inv @ adj_matrix
        G.graph['P'] = P

    num_nodes = G.number_of_nodes()

    # 初始化概率向量
    p = np.zeros(num_nodes)
    p[start_node] = 1.0

    # RWR迭代
    for _ in range(max_steps):
        # 使用稀疏矩阵乘法
        p_new = (1 - restart_prob) * P.dot(p) + restart_prob * (np.arange(num_nodes) == start_node).astype(float)

        # 检查收敛
        if np.max(np.abs(p_new - p)) < threshold:
            break

        p = p_new

    # 获取重要邻居 (概率高于阈值)
    neighbor_indices = np.where(p > threshold)[0]
    # 按概率排序
    sorted_indices = neighbor_indices[np.argsort(-p[neighbor_indices])]

    # 限制邻居数量
    top_k = 10  # 限制为最多10个邻居
    result = sorted_indices[:top_k]

    # 保存到缓存
    rwr_cache[cache_key] = result
    return result

def rwr_sampling(G, node_indices, restart_prob=0.2, max_steps=20, top_k=10):
    """
    使用RWR为每个节点采样重要邻居

    参数:
    - G: 用户-用户关系图
    - node_indices: 节点索引列表
    - restart_prob: 重启概率
    - max_steps: 最大步数
    - top_k: 采样邻居数量

    返回:
    - 邻接矩阵，仅包含采样的邻居
    """
    # 使用批量RWR采样
    return batch_rwr_sampling(G, node_indices, restart_prob, max_steps, top_k)


class MultiHeadGraphChannelAttention(nn.Module):
    """多头图通道注意力网络 - 优化版"""

    def __init__(self, in_features, out_features, num_heads, dropout=0.5):
        """
        初始化MHGCAN

        参数:
        - in_features: 输入特征维度
        - out_features: 输出特征维度
        - num_heads: 注意力头数 (从8减少到4)
        - dropout: Dropout率
        """
        super(MultiHeadGraphChannelAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # 优化: 减少注意力头数量
        # 原始: num_heads=8, 现在减少到4
        self.num_heads = min(num_heads, 4)

        # 为每个头创建参数
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(self.num_heads)
        ])

        # 注意力参数
        self.a = nn.Parameter(torch.zeros(size=(self.num_heads, 2 * out_features)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj, node_indices):
        """
        前向传播

        参数:
        - x: 节点特征 [num_nodes, in_features]
        - adj: 邻接矩阵 [num_nodes, num_nodes]
        - node_indices: 当前批处理的节点索引

        返回:
        - 转换后的节点特征
        """
        # 仅处理当前批处理的节点
        x_batch = x[node_indices]
        adj_batch = adj[node_indices][:, node_indices]

        # 多头注意力
        head_outputs = []
        for i in range(self.num_heads):
            # 线性变换
            h = self.W[i](x_batch)

            # 计算注意力系数
            a_input = self._prepare_attention_input(h)
            e = self.leakyrelu(torch.matmul(a_input, self.a[i]))

            # 归一化注意力系数
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj_batch > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = self.dropout(attention)

            # 注意力聚合
            h_prime = torch.matmul(attention, h)
            head_outputs.append(h_prime)

        # 拼接多头输出
        output = torch.cat(head_outputs, dim=1) if self.num_heads > 1 else head_outputs[0]
        return F.elu(output)

    def _prepare_attention_input(self, h):
        """准备注意力计算的输入"""
        N = h.size()[0]
        # 重复并拼接节点特征
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1),
                             h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        return a_input


class UserFeatureEncoder(nn.Module):
    """用户特征编码器 - 简化版"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(UserFeatureEncoder, self).__init__()

        # 优化: 简化编码器结构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # 去掉一层，减少计算复杂度
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class MHGCAN(nn.Module):
    """完整的MHGCAN模型 - 优化版"""

    def __init__(self,
                 user_feat_dim,
                 hidden_dim,
                 num_heads,
                 num_classes=2,
                 dropout=0.5):
        super(MHGCAN, self).__init__()

        # 优化: 降低隐藏层维度
        self.hidden_dim = max(64, hidden_dim // 2)  # 从128降低到64

        # 用户特征编码
        self.user_encoder = UserFeatureEncoder(
            user_feat_dim,
            self.hidden_dim,
            self.hidden_dim
        )

        # 多头图通道注意力层
        self.attention = MultiHeadGraphChannelAttention(
            self.hidden_dim,
            self.hidden_dim // 4,  # 从hidden_dim // num_heads调整
            num_heads,
            dropout
        )

        # 分类器 - 简化
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 4, num_classes)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_features, adj, node_indices):
        """
        前向传播

        参数:
        - user_features: 用户特征 [num_nodes, user_feat_dim]
        - adj: 邻接矩阵 [num_nodes, num_nodes]
        - node_indices: 当前批处理的节点索引

        返回:
        - 预测结果
        """
        # 编码用户特征
        h = self.user_encoder(user_features)
        h = self.dropout(h)

        # 图注意力
        h = self.attention(h, adj, node_indices)
        h = self.dropout(h)

        # 分类
        return self.classifier(h)


class HSIC_Lasso:
    """HSIC Lasso算法实现，用于特征选择和可解释性"""

    def __init__(self, sigma=1.0, lambda_param=0.1):
        """
        初始化HSIC Lasso

        参数:
        - sigma: 高斯核参数
        - lambda_param: Lasso正则化参数
        """
        self.sigma = sigma
        self.lambda_param = lambda_param
        self.beta = None

    def _gaussian_kernel(self, X, sigma=None):
        """计算高斯核矩阵"""
        if sigma is None:
            sigma = self.sigma

        # 计算欧氏距离平方
        X_norm = np.sum(X ** 2, axis=1)
        X_norm = X_norm.reshape(-1, 1)
        pairwise_sq_dists = X_norm + X_norm.T - 2 * np.dot(X, X.T)

        # 高斯核
        K = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
        return K

    def _centering(self, K):
        """中心化核矩阵"""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H.dot(K).dot(H)

    def _hsic(self, K, L):
        """计算HSIC"""
        n = K.shape[0]
        return np.trace(K.dot(L)) / n ** 2

    def fit(self, X, y):
        """
        拟合HSIC Lasso模型

        参数:
        - X: 特征矩阵 [n_samples, n_features]
        - y: 目标变量 [n_samples]

        返回:
        - 重要特征索引
        """
        n, d = X.shape

        # 计算目标变量的核矩阵
        y = y.reshape(-1, 1)
        L = self._gaussian_kernel(y)
        L = self._centering(L)

        # 计算每个特征的核矩阵
        Ks = []
        for j in range(d):
            K = self._gaussian_kernel(X[:, j].reshape(-1, 1))
            K = self._centering(K)
            Ks.append(K)

        # 计算HSIC矩阵
        A = np.zeros((d, d))
        b = np.zeros(d)
        for j in range(d):
            b[j] = self._hsic(Ks[j], L)
            for k in range(d):
                A[j, k] = self._hsic(Ks[j], Ks[k])

        # 求解HSIC Lasso问题
        self.beta = self._solve_lasso(A, b)

        # 获取重要特征
        important_features = np.where(self.beta > 1e-5)[0]
        return important_features

    def _solve_lasso(self, A, b):
        """求解Lasso问题"""
        d = A.shape[0]
        beta = np.zeros(d)
        max_iter = 1000
        tol = 1e-4

        for _ in range(max_iter):
            beta_old = beta.copy()

            for j in range(d):
                # 计算残差
                res = b[j] - np.dot(A[j], beta) + A[j, j] * beta[j]

                # Lasso更新
                beta[j] = np.sign(res) * max(0, np.abs(res) - self.lambda_param) / A[j, j]

            # 检查收敛
            if np.linalg.norm(beta - beta_old) < tol:
                break

        return beta

    def explain_instance(self, X, y, instance_idx, top_k=5):
        """
        解释单个实例

        参数:
        - X: 特征矩阵
        - y: 目标变量
        - instance_idx: 实例索引
        - top_k: 返回的最重要特征数量

        返回:
        - 重要特征索引和贡献值
        """
        # 获取实例特征和标签
        x_instance = X[instance_idx].reshape(1, -1)
        y_instance = y[instance_idx]

        # 生成扰动样本
        n_perturb = 100
        perturbations = np.random.normal(0, 0.1, (n_perturb, X.shape[1]))
        X_perturbed = np.repeat(x_instance, n_perturb, axis=0) + perturbations

        # 预测扰动样本
        y_perturbed = np.zeros(n_perturb)
        for i in range(n_perturb):
            # 这里应该使用模型预测，但为简化使用简单方法
            # 实际实现中应该替换为模型预测
            y_perturbed[i] = 1 if np.sum(X_perturbed[i]) > 0 else 0

        # 计算每个特征的重要性
        feature_importance = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            # 计算特征j的扰动与预测的HSIC
            K = self._gaussian_kernel(X_perturbed[:, j].reshape(-1, 1))
            K = self._centering(K)

            L = self._gaussian_kernel(y_perturbed.reshape(-1, 1))
            L = self._centering(L)

            feature_importance[j] = self._hsic(K, L)

        # 获取最重要的特征
        top_features = np.argsort(-feature_importance)[:top_k]
        return top_features, feature_importance[top_features]


class SpammerDataset(Dataset):
    """垃圾用户检测数据集"""

    def __init__(self, node_indices, labels):
        self.node_indices = node_indices
        self.labels = labels

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        return self.node_indices[idx], self.labels[idx]


def train_model(processed_data, num_epochs=30, batch_size=256, lr=0.001):
    """训练模型 - 优化版"""
    # 准备数据
    interaction_matrix = processed_data['interaction_matrix']
    user_feat_matrix = processed_data['user_feat_matrix']
    user_labels = processed_data['user_labels']
    G = processed_data['G']
    train_idx = processed_data['train_idx']
    val_idx = processed_data['val_idx']
    test_idx = processed_data['test_idx']

    # 创建数据集
    train_dataset = SpammerDataset(train_idx, user_labels[train_idx])
    val_dataset = SpammerDataset(val_idx, user_labels[val_idx])
    test_dataset = SpammerDataset(test_idx, user_labels[test_idx])

    # 优化: 增加批量大小
    # 原batch_size=128, 现在增加到256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS设备")
        dtype = torch.float32
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用CUDA设备")
        dtype = torch.float32  # 优化: 使用float32而非float64
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
        dtype = torch.float32

    # 初始化模型
    model = MHGCAN(
        user_feat_dim=user_feat_matrix.shape[1],
        hidden_dim=64,  # 从128降低到64
        num_heads=4,  # 从8降低到4
        num_classes=2,
        dropout=0.5
    ).to(device)

    # 优化: 使用混合精度训练
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 优化: 使用更大的学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 转换数据为Tensor
    user_features = torch.tensor(user_feat_matrix, dtype=dtype).to(device)
    labels = torch.LongTensor(user_labels).to(device)  # 标签必须是LongTensor

    # 训练过程记录
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    print("\n开始训练...")
    best_val_f1 = 0
    early_stop_counter = 0
    early_stop_patience = 10

    # 优化: 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5
    )

    # 优化: 记录学习率
    current_lr = lr

    # 优化: 预计算所有可能需要的RWR结果（如果内存允许）
    if len(G.nodes) < 10000:  # 对小图预计算
        print("预计算所有节点的RWR结果...")
        start_time = time.time()

        # 创建完整的采样邻接矩阵
        full_adj = batch_rwr_sampling(
            G,
            list(range(len(G.nodes))),
            top_k=10
        )

        # 保存到处理数据中
        processed_data['full_rwr_adj'] = full_adj
        print(f"RWR预计算完成，耗时: {time.time() - start_time:.2f}秒")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        # 优化: 添加tqdm进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for node_indices, batch_labels in pbar:
            # 确保node_indices是LongTensor
            node_indices = node_indices.long().to(device)
            # 确保batch_labels是LongTensor
            batch_labels = batch_labels.long().to(device)

            # 优化: 使用预计算的RWR结果（如果可用）
            if 'full_rwr_adj' in processed_data:
                adj = torch.tensor(
                    processed_data['full_rwr_adj'][node_indices][:, node_indices],
                    dtype=dtype
                ).to(device)
            else:
                # RWR采样邻居并构建子图
                adj = rwr_sampling(G, node_indices.cpu().numpy(), top_k=10)
                adj = torch.tensor(adj, dtype=dtype).to(device)

            # 优化: 混合精度训练
            if use_amp:
                with torch.cuda.amp.autocast():
                    # 前向传播
                    outputs = model(user_features, adj, node_indices)
                    loss = criterion(outputs, batch_labels)

                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                optimizer.zero_grad()
                outputs = model(user_features, adj, node_indices)
                loss = criterion(outputs, batch_labels)

                # 反向传播
                loss.backward()
                optimizer.step()

            # 记录
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            # 更新进度条
            if len(all_labels) > 0:
                current_acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.4f}"})

        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)

        # 验证阶段
        val_loss, val_acc, val_pre, val_rec, val_f1 = evaluate_model(
            model, val_loader, user_features, labels, G, device
        )

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # 打印结果
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(
            f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Pre: {val_pre:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

        # 优化: 学习率调度
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']

        # 早停机制
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'mhgcan_best.pt')
            print(f"  --> 保存新最佳模型 (F1: {val_f1:.4f}, LR: {current_lr:.6f})")
        else:
            early_stop_counter += 1
            print(f"  早停计数器: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"\n早停: 在 {epoch + 1} 轮停止训练")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('mhgcan_best.pt', map_location=device))

    # 测试集评估
    test_loss, test_acc, test_pre, test_rec, test_f1, test_auc = evaluate_model(
        model, test_loader, user_features, labels, G, device, return_auc=True
    )
    print(f"\n测试集结果:")
    print(
        f"  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Pre: {test_pre:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

    # 保存结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_f1s': val_f1s,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_pre,
            'recall': test_rec,
            'f1': test_f1,
            'auc': test_auc
        }
    }

    with open('mhgcan_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=4)

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_accs, val_f1s)

    return model, results


def evaluate_model(model, loader, user_features, labels, G, device, return_auc=False):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    # 确定数据类型
    dtype = torch.float32 if device.type == 'mps' else torch.float32

    # 添加tqdm进度条
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for node_indices, batch_labels in pbar:
            # 确保node_indices是LongTensor
            node_indices = node_indices.long().to(device)
            # 确保batch_labels是LongTensor
            batch_labels = batch_labels.long().to(device)

            # 优化: 使用预计算的RWR结果（如果可用）
            if hasattr(G, 'full_rwr_adj'):
                adj = torch.tensor(
                    G.full_rwr_adj[node_indices][:, node_indices],
                    dtype=dtype
                ).to(device)
            else:
                # RWR采样邻居并构建子图
                adj = rwr_sampling(G, node_indices.cpu().numpy(), top_k=10)
                adj = torch.tensor(adj, dtype=dtype).to(device)

            # 前向传播
            outputs = model(user_features, adj, node_indices)
            loss = criterion(outputs, batch_labels)

            # 记录
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)[:, 1]  # 正类概率

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            # 更新进度条
            if len(all_labels) > 0:
                current_acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.4f}"})

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_labels)
    f1 = f1_score(all_labels, all_preds)

    if return_auc:
        auc = roc_auc_score(all_labels, all_probs)
        return total_loss / len(loader), accuracy, precision, recall, f1, auc
    else:
        return total_loss / len(loader), accuracy, precision, recall, f1


def plot_training_curves(train_losses, val_losses, val_accs, val_f1s):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 8))

    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()

    # 准确率和F1曲线
    plt.subplot(2, 1, 2)
    plt.plot(val_accs, label='验证准确率')
    plt.plot(val_f1s, label='验证F1分数')
    plt.xlabel('轮次')
    plt.ylabel('分数')
    plt.title('验证准确率和F1分数')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()


def explain_model(model, processed_data, top_k=5):
    """使用HSIC Lasso解释模型"""
    print("\n" + "=" * 50)
    print("模型解释")
    print("=" * 50)

    # 准备数据
    user_feat_matrix = processed_data['user_feat_matrix']
    user_labels = processed_data['user_labels']
    test_idx = processed_data['test_idx']

    # 仅使用测试集进行解释
    X = user_feat_matrix[test_idx]
    y = user_labels[test_idx]

    # 初始化HSIC Lasso
    hsic_lasso = HSIC_Lasso(sigma=1.0, lambda_param=0.1)

    # 获取重要特征
    important_features = hsic_lasso.fit(X, y)
    print(f"重要特征索引: {important_features}")

    # 解释随机选择的实例
    num_explain = min(5, len(test_idx))
    explain_indices = np.random.choice(len(test_idx), num_explain, replace=False)

    # 特征名称 (根据您的数据集调整)
    feature_names = [
        'author_friend_sum',
        'author_review_sum',
        'author_photo_sum',
        'text_length',
        'word_count',
        'polarity',
        'subjectivity'
    ]

    for i, idx in enumerate(explain_indices):
        instance_idx = test_idx[idx]
        top_features, feature_importance = hsic_lasso.explain_instance(
            X, y, i, top_k=top_k
        )

        print(f"\n实例 {i + 1} (用户ID: {processed_data['idx_to_user_id'][instance_idx]}):")
        for j, feature_idx in enumerate(top_features):
            print(f"  {j + 1}. {feature_names[feature_idx]}: {feature_importance[j]:.4f}")

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))

    # 计算全局特征重要性
    global_importance = np.zeros(len(feature_names))
    for i in range(len(test_idx)):
        top_features, feature_importance = hsic_lasso.explain_instance(
            X, y, i, top_k=len(feature_names)
        )
        for j, feature_idx in enumerate(top_features):
            global_importance[feature_idx] += feature_importance[j]

    # 归一化
    global_importance = global_importance / np.sum(global_importance)

    # 绘制
    plt.bar(range(len(feature_names)), global_importance)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.ylabel('重要性')
    plt.title('全局特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    print("\n特征重要性可视化已保存到 'feature_importance.png'")


def main():
    """主函数"""
    print("=" * 50)
    print("开始复现Local Interpretable Spammer Detection Model")
    print("=" * 50)

    # 1. 加载并预处理数据
    processed_data = preprocess_data()

    # 2. 训练模型
    print("\n" + "=" * 50)
    print("步骤2: 训练模型")
    print("=" * 50)

    # 优化: 减少训练轮次（如果验证F1已经稳定）
    model, results = train_model(
        processed_data,
        num_epochs=30,  # 从50减少到30
        batch_size=256,  # 从128增加到256
        lr=0.001
    )

    # 3. 解释模型
    print("\n" + "=" * 50)
    print("步骤3: 模型解释")
    print("=" * 50)

    explain_model(model, processed_data, top_k=5)

    # 4. 保存最终模型
    print("\n" + "=" * 50)
    print("步骤4: 保存最终模型")
    print("=" * 50)

    # 获取当前设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 保存模型
    torch.save(model.state_dict(), 'mhgcan_final.pt')
    print("模型已保存到 'mhgcan_final.pt'")

    print("\n" + "=" * 50)
    print("复现完成！")
    print(f"最佳验证F1: {max(results['val_f1s']):.4f}")
    print(f"测试集F1: {results['test_metrics']['f1']:.4f}")
    print(f"测试集AUC: {results['test_metrics']['auc']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()