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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    RobertaTokenizer,
    RobertaModel
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")


# 1. 数据预处理
def preprocess_text(text):
    """文本预处理函数"""
    if pd.isna(text) or text == '':
        return "empty review"
    # 移除特殊字符和多余空格
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_data():
    """加载并预处理数据"""
    # 加载数据
    data_path = "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv"
    df = pd.read_csv(data_path)

    print(f"原始数据形状: {df.shape}")
    print(f"真实评论比例: {df['is_recommended'].mean():.2f}")

    # 确保目标变量无缺失
    assert df['is_recommended'].isnull().sum() == 0, "is_recommended contains missing values"

    # 文本预处理
    df['processed_content'] = df['content'].apply(preprocess_text)

    # 用户特征处理
    user_features = ['author_friend_sum', 'author_review_sum', 'author_photo_sum']
    user_feat_matrix = df[user_features].fillna(0).values

    # 标准化用户特征
    scaler = StandardScaler()
    user_feat_matrix = scaler.fit_transform(user_feat_matrix)
    df[user_features] = user_feat_matrix

    # 商家特征处理
    def parse_review_counts(rating_str):
        """解析商家评分统计"""
        if pd.isna(rating_str) or rating_str == '':
            return [0, 0, 0, 0, 0]
        try:
            # 尝试解析JSON格式
            counts = json.loads(rating_str.replace("'", "\""))
            return [counts.get(str(i), 0) for i in range(1, 6)]
        except:
            # 简单分割格式
            parts = str(rating_str).strip('[]').split(',')
            return [int(p.strip()) for p in parts[:5]] if len(parts) >= 5 else [0, 0, 0, 0, 0]

    df['biz_review_counts'] = df['biz_reviewCountsByRating'].apply(parse_review_counts)
    biz_review_counts = np.array(df['biz_review_counts'].tolist())

    # 其他商家特征
    biz_other_features = df[['biz_reviewCount', 'biz_rating']].fillna(0).values
    # 标准化
    biz_other_features = StandardScaler().fit_transform(biz_other_features)

    # 合并商家特征
    biz_feat_matrix = np.hstack([biz_review_counts, biz_other_features])
    for i in range(5):
        df[f'biz_rating_{i + 1}'] = biz_review_counts[:, i]
    df['biz_reviewCount_norm'] = biz_other_features[:, 0]
    df['biz_rating_norm'] = biz_other_features[:, 1]

    # 划分数据集 - 仅训练集用于知识生成
    X = df.index.values
    y = df['is_recommended'].values
    train_idx, test_idx, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_idx, y_train, test_size=0.125, random_state=42, stratify=y_train
    )

    print(f"训练集: {len(train_idx)} ({np.mean(y_train):.2f} 真实), 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")

    return df, train_idx, val_idx, test_idx


# 2. 本地部署Qwen-8B生成知识
class QwenKnowledgeGenerator:
    """使用本地Qwen-8B生成知识的类"""

    def __init__(self, model_name="Qwen/Qwen3-8B", cache_dir="./qwen_cache"):
        """
        初始化Qwen知识生成器

        参数:
        - model_name: Qwen模型名称
        - cache_dir: 缓存目录，避免重复生成
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 检查是否有足够显存
        self.quantize = True  # 默认使用量化
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            # 如果显存大于20GB，可以考虑不量化
            if total_memory > 20:
                self.quantize = False

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载Qwen模型"""
        print(f"加载Qwen模型: {self.model_name} (4-bit量化: {self.quantize})")

        # 4-bit量化配置
        bnb_config = None
        if self.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False
        )

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("Qwen模型加载完成")

    def _create_prompt(self, review):
        """创建Qwen的提示模板"""
        system_prompt = (
            "你是一个专业的餐厅评论真实性分析专家。"
            "请严格按照要求以JSON格式返回结果，不要包含其他解释。"
        )

        user_prompt = f"""
        请分析以下餐厅评论的真实性。请提供：
        1. 评论的关键事实点
        2. 可能的矛盾或不一致之处
        3. 该评论是否符合常见餐厅体验
        4. 该评论是否包含可疑的过度赞美或贬低

        评论内容: "{review}"

        请以严格的JSON格式返回结果，包含以下字段:
        - key_facts: 评论中的关键事实点列表
        - inconsistencies: 可能的矛盾或不一致之处列表
        - consistency_with_common_experience: 与常见餐厅体验的一致性评分(0-1)
        - suspicious_elements: 可疑元素列表
        - authenticity_score: 真实性评分(0-1)

        注意: 
        1. 仅返回JSON，不要包含其他解释
        2. authenticity_score应基于评论的可信度，真实评论接近1，虚假评论接近0
        3. 保持客观，不要过度主观判断
        """
        return system_prompt, user_prompt

    def _generate_response(self, system_prompt, user_prompt, max_new_tokens=300):
        """生成Qwen响应"""
        # 构建对话
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 应用Qwen的chat模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码响应
        response = self.tokenizer.decode(
            outputs[0][len(model_inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        return response

    def _extract_json(self, response):
        """从响应中提取JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从响应中提取JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
        return None

    def generate_knowledge(self, reviews, batch_size=2, cache_file="train_knowledge.json"):
        """
        生成知识

        参数:
        - reviews: 评论列表
        - batch_size: 批处理大小
        - cache_file: 缓存文件名

        返回:
        - knowledge_list: 知识列表
        """
        cache_path = os.path.join(self.cache_dir, cache_file)

        # 检查缓存
        if os.path.exists(cache_path):
            print(f"从缓存 {cache_path} 加载知识...")
            with open(cache_path, 'r') as f:
                return json.load(f)

        print(f"缓存 {cache_path} 不存在，开始生成知识...")
        knowledge_list = []
        total = len(reviews)

        # 处理每条评论
        for i, review in enumerate(tqdm(reviews, desc="生成知识")):
            # 检查是否为空
            if not review.strip() or review == "empty review":
                knowledge_list.append({
                    "key_facts": [],
                    "inconsistencies": ["empty review"],
                    "consistency_with_common_experience": 0.5,
                    "suspicious_elements": ["empty review"],
                    "authenticity_score": 0.5
                })
                continue

            # 生成知识
            system_prompt, user_prompt = self._create_prompt(review)
            response = self._generate_response(system_prompt, user_prompt)

            # 提取JSON
            knowledge = self._extract_json(response)

            # 如果提取失败，使用默认值
            if knowledge is None:
                knowledge = {
                    "key_facts": [],
                    "inconsistencies": [f"LLM响应解析失败: {response[:100]}..."],
                    "consistency_with_common_experience": 0.5,
                    "suspicious_elements": ["LLM响应解析失败"],
                    "authenticity_score": 0.5
                }

            knowledge_list.append(knowledge)

            # 保存进度（每10条）
            if (i + 1) % 10 == 0 or i == total - 1:
                with open(cache_path, 'w') as f:
                    json.dump(knowledge_list, f)

        print(f"知识生成完成，已保存到 {cache_path}")
        return knowledge_list


# 3. LEKD模型实现
class TextEncoder(nn.Module):
    """文本编码器 - 使用RoBERTa"""

    def __init__(self, pretrained_model='roberta-base'):
        super(TextEncoder, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        self.hidden_size = self.roberta.config.hidden_size
        self.device = next(self.roberta.parameters()).device  # 获取模型初始设备

        # 冻结RoBERTa参数（可选）
        for param in self.roberta.parameters():
            param.requires_grad = False

    def to(self, device):
        """确保模型移动到指定设备"""
        self.roberta = self.roberta.to(device)
        self.device = device
        return super().to(device)

    def forward(self, texts):
        """编码文本"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        # 使用模型所在设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.roberta(**inputs)

        # 使用[CLS] token作为句子表示
        return outputs.last_hidden_state[:, 0, :]


class KnowledgeProcessor(nn.Module):
    """处理从LLM提取的知识"""

    def __init__(self, text_encoder, hidden_size=256):
        super(KnowledgeProcessor, self).__init__()
        self.text_encoder = text_encoder
        self.hidden_size = hidden_size

        # 知识特征处理网络
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(text_encoder.hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 互信息瓶颈组件
        self.mi_bottleneck = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

    def to(self, device):
        """移动模型到指定设备"""
        self.text_encoder.to(device)
        self.knowledge_encoder = self.knowledge_encoder.to(device)
        self.mi_bottleneck = self.mi_bottleneck.to(device)
        return super().to(device)

    def forward(self, reviews, knowledge_list):
        """处理知识"""
        # 确保文本编码器在正确设备上
        device = next(self.knowledge_encoder.parameters()).device
        self.text_encoder.to(device)

        # 编码原始评论
        review_embeddings = self.text_encoder(reviews)

        # 创建知识增强的评论表示
        knowledge_enhanced = []
        for i, (review, knowledge) in enumerate(zip(reviews, knowledge_list)):
            # 将知识转换为文本
            knowledge_text = self._knowledge_to_text(knowledge)

            # 编码知识文本
            with torch.no_grad():
                knowledge_embedding = self.text_encoder([knowledge_text])
                # 确保知识嵌入在相同设备上
                knowledge_embedding = knowledge_embedding.to(device)

            # 合并原始评论和知识
            combined = torch.cat([review_embeddings[i], knowledge_embedding[0]], dim=0)
            knowledge_enhanced.append(combined)

        knowledge_enhanced = torch.stack(knowledge_enhanced).to(device)

        # 通过知识编码器
        z_B = self.knowledge_encoder(knowledge_enhanced)

        # 互信息瓶颈
        z_tB = self.mi_bottleneck(z_B)

        return z_B, z_tB

    def _knowledge_to_text(self, knowledge):
        """将知识字典转换为文本"""
        facts = ", ".join(knowledge.get('key_facts', []))
        inconsistencies = ", ".join(knowledge.get('inconsistencies', []))
        suspicious = ", ".join(knowledge.get('suspicious_elements', []))

        text = f"Key facts: {facts}. Inconsistencies: {inconsistencies}. Suspicious elements: {suspicious}."
        return text


class GraphSemanticAlignment(nn.Module):
    """图语义感知特征对齐模块"""

    def __init__(self, feature_dim, num_heads=4):
        super(GraphSemanticAlignment, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # 多头注意力
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)
        self.W_o = nn.Linear(feature_dim, feature_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)

    def to(self, device):
        """移动模型到指定设备"""
        self.W_q = self.W_q.to(device)
        self.W_k = self.W_k.to(device)
        self.W_v = self.W_v.to(device)
        self.W_o = self.W_o.to(device)
        self.ffn = self.ffn.to(device)
        self.layer_norm1 = self.layer_norm1.to(device)
        self.layer_norm2 = self.layer_norm2.to(device)
        return super().to(device)

    def forward(self, text_features, knowledge_features):
        """特征对齐"""
        batch_size = text_features.size(0)

        # 多头注意力
        Q = self.W_q(text_features).view(batch_size, self.num_heads, self.head_dim)
        K = self.W_k(knowledge_features).view(batch_size, self.num_heads, self.head_dim)
        V = self.W_v(knowledge_features).view(batch_size, self.num_heads, self.head_dim)

        # 计算注意力分数
        attn_scores = torch.einsum('bhd,bhd->bh', Q, K) / (self.head_dim ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # 应用注意力
        attn_output = torch.einsum('bh,bhd->bhd', attn_scores, V)
        attn_output = attn_output.contiguous().view(batch_size, -1)

        # 输出投影
        attn_output = self.W_o(attn_output)

        # 残差连接和层归一化
        aligned_features = self.layer_norm1(text_features + attn_output)

        # 前馈网络
        ffn_output = self.ffn(aligned_features)

        # 残差连接和层归一化
        aligned_features = self.layer_norm2(aligned_features + ffn_output)

        return aligned_features


class LEKDModel(nn.Module):
    """LEKD模型实现"""

    def __init__(self,
                 text_encoder,
                 user_feature_dim,
                 biz_feature_dim,
                 hidden_size=256,
                 alpha=0.1):
        super(LEKDModel, self).__init__()
        self.text_encoder = text_encoder
        self.hidden_size = hidden_size
        self.alpha = alpha  # 互信息损失系数

        # 知识处理器
        self.knowledge_processor = KnowledgeProcessor(text_encoder, hidden_size)

        # 用户特征处理
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # 商家特征处理
        self.biz_encoder = nn.Sequential(
            nn.Linear(biz_feature_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # 图语义感知特征对齐
        self.gsa = GraphSemanticAlignment(hidden_size)

        # 融合模块
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_size // 2, 1)

    def to(self, device):
        """移动模型到指定设备"""
        self.text_encoder.to(device)
        self.knowledge_processor.to(device)
        self.user_encoder = self.user_encoder.to(device)
        self.biz_encoder = self.biz_encoder.to(device)
        self.gsa = self.gsa.to(device)
        self.fusion = self.fusion.to(device)
        self.classifier = self.classifier.to(device)
        return self

    def forward(self, reviews, user_features, biz_features, knowledge_list, labels=None):
        """
        前向传播

        参数:
        - reviews: 评论列表
        - user_features: 用户特征 [batch_size, user_feature_dim]
        - biz_features: 商家特征 [batch_size, biz_feature_dim]
        - knowledge_list: 知识列表
        - labels: 标签 [batch_size]

        返回:
        - probs: 预测概率
        - losses: 损失字典
        """
        # 1. 获取知识增强表示
        z_B, z_tB = self.knowledge_processor(reviews, knowledge_list)

        # 2. 处理用户特征
        user_features = user_features.to(self.user_encoder[0].weight.device)
        user_rep = self.user_encoder(user_features)

        # 3. 处理商家特征
        biz_features = biz_features.to(self.biz_encoder[0].weight.device)
        biz_rep = self.biz_encoder(biz_features)

        # 4. 图语义感知特征对齐
        aligned_text = self.gsa(z_tB, user_rep)

        # 5. 融合所有特征
        combined = torch.cat([aligned_text, user_rep, biz_rep], dim=1)
        fused = self.fusion(combined)

        # 6. 分类
        logits = self.classifier(fused)
        probs = torch.sigmoid(logits)

        # 7. 计算损失
        losses = {}
        if labels is not None:
            # 分类损失
            labels = labels.float().unsqueeze(1).to(logits.device)
            classification_loss = F.binary_cross_entropy(probs, labels)
            losses['classification'] = classification_loss

            # 互信息蒸馏损失 (KL散度)
            teacher_probs = self._get_teacher_predictions(knowledge_list)
            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.softmax(teacher_probs, dim=1),
                reduction='batchmean'
            )
            losses['kl'] = kl_loss

            # 总损失
            total_loss = classification_loss + self.alpha * kl_loss
            losses['total'] = total_loss

        return probs, losses

    def _get_teacher_predictions(self, knowledge_list):
        """从知识中获取"教师"模型的预测"""
        teacher_probs = []
        for knowledge in knowledge_list:
            # 从知识中提取真实性评分
            authenticity_score = knowledge.get('authenticity_score', 0.5)
            # 确保在[0,1]范围内
            authenticity_score = max(0.0, min(1.0, authenticity_score))
            teacher_probs.append([authenticity_score])

        teacher_probs = torch.FloatTensor(teacher_probs).to(self.classifier.weight.device)
        return teacher_probs


# 4. 数据集和训练
class ReviewDataset(Dataset):
    """评论数据集"""

    def __init__(self, df, indices, knowledge_list=None):
        self.df = df.loc[indices].reset_index(drop=True)
        self.knowledge_list = knowledge_list

        # 修复：移除author_char_id，它是一个字符串ID，不是数值特征
        # 原始错误代码: self.user_features = ['author_friend_sum', 'author_review_sum', 'author_photo_sum', 'author_char_id']
        self.user_features = ['author_friend_sum', 'author_review_sum', 'author_photo_sum']

        # 商家特征列
        self.biz_features = [
            'biz_rating_1', 'biz_rating_2', 'biz_rating_3',
            'biz_rating_4', 'biz_rating_5', 'biz_reviewCount_norm', 'biz_rating_norm'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 获取特征
        # 确保只选择数值特征
        user_features = row[self.user_features].values.astype(np.float32)
        biz_features = row[self.biz_features].values.astype(np.float32)

        # 获取标签
        label = row['is_recommended']

        # 获取知识（如果提供）
        knowledge = None
        if self.knowledge_list is not None:
            knowledge = self.knowledge_list[idx]

        return {
            'review': row['processed_content'],
            'user_features': user_features,
            'biz_features': biz_features,
            'label': label,
            'knowledge': knowledge
        }


def collate_fn(batch):
    """自定义批处理函数"""
    reviews = [item['review'] for item in batch]
    user_features = torch.tensor([item['user_features'] for item in batch])
    biz_features = torch.tensor([item['biz_features'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)

    knowledge_list = [item['knowledge'] for item in batch]

    return {
        'reviews': reviews,
        'user_features': user_features,
        'biz_features': biz_features,
        'labels': labels,
        'knowledge_list': knowledge_list
    }


def train_model(df, train_idx, val_idx, test_idx, train_knowledge, val_knowledge, test_knowledge):
    """训练模型"""
    # 创建数据集
    train_dataset = ReviewDataset(df, train_idx, train_knowledge)
    val_dataset = ReviewDataset(df, val_idx, val_knowledge)
    test_dataset = ReviewDataset(df, test_idx, test_knowledge)

    # 创建数据加载器
    batch_size = 8  # 根据显存调整
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
    text_encoder = TextEncoder('roberta-base')
    model = LEKDModel(
        text_encoder,
        user_feature_dim=len(train_dataset.user_features),
        biz_feature_dim=len(train_dataset.biz_features),
        hidden_size=256
    )

    # 关键修复：确保将整个模型移动到设备上
    model = model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
        weight_decay=1e-4
    )
    last_lr = optimizer.param_groups[0]['lr']
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2, factor=0.5
    )

    # 训练参数
    num_epochs = 5
    best_val_f1 = 0
    early_stop_counter = 0
    early_stop_patience = 5

    # 记录训练过程
    train_losses = []
    val_losses = []
    val_f1s = []

    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            # 准备数据 - 确保移动到正确设备
            reviews = batch['reviews']
            user_features = batch['user_features'].to(device)
            biz_features = batch['biz_features'].to(device)
            labels = batch['labels'].to(device)
            knowledge_list = batch['knowledge_list']

            # 前向传播
            optimizer.zero_grad()
            probs, losses = model(
                reviews, user_features, biz_features, knowledge_list, labels
            )

            # 反向传播
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 记录损失
            total_train_loss += losses['total'].item()

            # 记录预测结果
            preds = (probs > 0.5).cpu().detach().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

        # 计算训练指标
        train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds)

        # 验证阶段
        val_loss, val_acc, val_pre, val_rec, val_f1 = evaluate_model(model, val_loader)

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        # 打印结果
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(
            f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Pre: {val_pre:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

        # 学习率调度
        scheduler.step(val_f1)

        # 手动检查学习率是否降低并打印信息
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < last_lr:
            print(f"  学习率已降低到 {current_lr:.8f}")
        last_lr = current_lr

        # 早停机制
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'lekdbest_model.pt')
            print("  --> 保存新最佳模型")
        else:
            early_stop_counter += 1
            print(f"  早停计数器: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"\n早停: 在 {epoch + 1} 轮停止训练")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('lekdbest_model.pt'))

    # 测试集评估
    test_loss, test_acc, test_pre, test_rec, test_f1 = evaluate_model(model, test_loader)
    print(f"\n测试集结果:")
    print(f"  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Pre: {test_pre:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")

    # 保存结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1s': val_f1s,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_pre,
            'recall': test_rec,
            'f1': test_f1
        }
    }

    with open('lekdbest_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_f1s)

    return model, results


def evaluate_model(model, loader):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估"):
            # 准备数据
            reviews = batch['reviews']
            user_features = batch['user_features'].to(device)
            biz_features = batch['biz_features'].to(device)
            labels = batch['labels'].to(device)
            knowledge_list = batch['knowledge_list']

            # 前向传播
            probs, losses = model(
                reviews, user_features, biz_features, knowledge_list, labels
            )

            # 记录损失
            total_loss += losses['total'].item()

            # 记录预测结果
            preds = (probs > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_labels)
    f1 = f1_score(all_labels, all_preds)

    return total_loss / len(loader), accuracy, precision, recall, f1


def plot_training_curves(train_losses, val_losses, val_f1s):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()

    # F1曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, label='验证F1')
    plt.xlabel('轮次')
    plt.ylabel('F1分数')
    plt.title('验证F1分数')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()


# 5. 主函数
def main():
    """主函数"""
    print("=" * 50)
    print("开始复现LEKD: 基于外部知识蒸馏的虚假评论检测")
    print("=" * 50)

    # 1. 加载并预处理数据
    df, train_idx, val_idx, test_idx = load_and_preprocess_data()

    # 2. 生成或加载知识
    print("\n" + "=" * 50)
    print("步骤2: 生成知识")
    print("=" * 50)

    # 检查是否已有知识
    train_knowledge_file = "./qwen_cache/train_knowledge.json"
    val_knowledge_file = "./qwen_cache/val_knowledge.json"
    test_knowledge_file = "./qwen_cache/test_knowledge.json"

    # 如果已有知识文件，直接加载
    if os.path.exists(train_knowledge_file):
        print(f"发现训练集知识文件: {train_knowledge_file}")
        with open(train_knowledge_file, 'r') as f:
            train_knowledge = json.load(f)
    else:
        # 初始化Qwen知识生成器
        qwen_generator = QwenKnowledgeGenerator()

        # 为训练集生成知识（仅训练集！）
        train_reviews = df.loc[train_idx, 'processed_content'].tolist()
        train_knowledge = qwen_generator.generate_knowledge(
            train_reviews,
            batch_size=2,
            cache_file="train_knowledge.json"
        )

    # 为验证集和测试集生成默认知识（不调用LLM）
    print("\n为验证集和测试集生成默认知识...")
    val_reviews = df.loc[val_idx, 'processed_content'].tolist()
    test_reviews = df.loc[test_idx, 'processed_content'].tolist()

    # 验证集知识（使用默认值）
    val_knowledge = [{
        "key_facts": [],
        "inconsistencies": [],
        "consistency_with_common_experience": 0.5,
        "suspicious_elements": [],
        "authenticity_score": 0.5
    } for _ in range(len(val_reviews))]

    # 测试集知识（使用默认值）
    test_knowledge = [{
        "key_facts": [],
        "inconsistencies": [],
        "consistency_with_common_experience": 0.5,
        "suspicious_elements": [],
        "authenticity_score": 0.5
    } for _ in range(len(test_reviews))]

    # 保存验证集和测试集知识
    with open(val_knowledge_file, 'w') as f:
        json.dump(val_knowledge, f)
    with open(test_knowledge_file, 'w') as f:
        json.dump(test_knowledge, f)

    # 3. 训练模型
    print("\n" + "=" * 50)
    print("步骤3: 训练模型")
    print("=" * 50)

    model, results = train_model(
        df, train_idx, val_idx, test_idx,
        train_knowledge, val_knowledge, test_knowledge
    )

    # 4. 保存最终模型
    print("\n" + "=" * 50)
    print("步骤4: 保存最终模型")
    print("=" * 50)

    torch.save(model.state_dict(), 'lekdbest_final_model.pt')
    print("模型已保存到 'lekdbest_final_model.pt'")

    print("\n" + "=" * 50)
    print("复现完成！")
    print(f"最佳验证F1: {max(results['val_f1s']):.4f}")
    print(f"测试集F1: {results['test_metrics']['f1']:.4f}")
    print("=" * 50)


def evaluate_existing_model():
    """评估已有模型（如果存在）"""
    if not os.path.exists('lekdbest_model.pt'):
        print("未找到已训练模型，跳过评估")
        return

    print("\n" + "=" * 50)
    print("评估已有模型")
    print("=" * 50)

    # 加载数据
    df, _, val_idx, test_idx = load_and_preprocess_data()

    # 加载知识
    with open('./qwen_cache/val_knowledge.json', 'r') as f:
        val_knowledge = json.load(f)
    with open('./qwen_cache/test_knowledge.json', 'r') as f:
        test_knowledge = json.load(f)

    # 创建数据集
    val_dataset = ReviewDataset(df, val_idx, val_knowledge)
    test_dataset = ReviewDataset(df, test_idx, test_knowledge)

    # 创建数据加载器
    batch_size = 8
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 初始化模型
    text_encoder = TextEncoder('roberta-base')
    model = LEKDModel(
        text_encoder,
        user_feature_dim=len(val_dataset.user_features),
        biz_feature_dim=len(val_dataset.biz_features),
        hidden_size=256
    )

    # 加载模型
    model.load_state_dict(torch.load('lekdbest_model.pt'))
    model.to(device)

    # 评估
    val_loss, val_acc, val_pre, val_rec, val_f1 = evaluate_model(model, val_loader)
    test_loss, test_acc, test_pre, test_rec, test_f1 = evaluate_model(model, test_loader)

    print(f"\n验证集结果:")
    print(f"  Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Pre: {val_pre:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

    print(f"\n测试集结果:")
    print(f"  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Pre: {test_pre:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")


if __name__ == "__main__":
    # 检查是否已有训练好的模型
    if os.path.exists('lekdbest_model.pt'):
        print("检测到已有训练模型，是否评估？(y/n)")
        choice = input().lower()
        if choice == 'y':
            evaluate_existing_model()
        else:
            main()
    else:
        main()

# 0.4235, Acc: 0.8155, Pre: 0.9375, Rec: 1.0000, F1: 0.8063