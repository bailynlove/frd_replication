import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor, CLIPModel, CLIPProcessor
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
from torch_geometric.nn import GATConv  # 使用GAT作为Graph Transformer的基础
from torch_geometric.data import Data, Batch
from collections import defaultdict

warnings.filterwarnings("ignore")

# --- 命令行参数 ---
parser = argparse.ArgumentParser(description='Train CR-FND Model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
args = parser.parse_args()

# --- 配置 ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "image_dir": "../spams_detection/spam_datasets/crawler/LA/images/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 20,
    "learning_rate": 1e-4,
    "bert_model": "bert-base-uncased",
    "vit_model": "google/vit-base-patch16-224-in21k",
    "clip_model": "openai/clip-vit-base-patch32",
    "hidden_dim": 256,
    "num_classes": 2,
    "best_model_path": "cr_fnd_best_model.pth"
}

print(f"使用设备: {CONFIG['device']}")


# --- 模型子模块 ---

class VAE(nn.Module):  # 简化的VAE用于主题特征提取
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        return mu, log_var


class GraphTransformer(nn.Module):  # 简化的Graph Transformer
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, out_dim, heads=heads)
        self.conv2 = GATConv(out_dim * heads, out_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(x, data.batch)


# --- 主模型 CR-FND ---
class CRFND(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.vit = ViTModel.from_pretrained(config["vit_model"])
        self.clip = CLIPModel.from_pretrained(config["clip_model"])

        # 语义特征提取器
        bert_dim = self.bert.config.hidden_size
        vit_dim = self.vit.config.hidden_size
        clip_dim = self.clip.config.text_config.hidden_size
        self.topic_vae = VAE(vocab_size, 512, config["hidden_dim"])

        # 投影层，统一维度
        self.text_proj = nn.Linear(bert_dim, config["hidden_dim"])
        self.image_proj = nn.Linear(vit_dim, config["hidden_dim"])
        self.clip_proj = nn.Linear(clip_dim, config["hidden_dim"])

        # 传播结构特征提取器
        self.graph_transformer = GraphTransformer(bert_dim, config["hidden_dim"])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config["hidden_dim"] * 3, config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config["hidden_dim"], config["num_classes"])
        )

    def get_semantic_features(self, text_ids, text_mask, images, bow_vector):
        text_feat = self.text_proj(
            self.bert(input_ids=text_ids, attention_mask=text_mask).last_hidden_state.mean(dim=1))
        image_feat = self.image_proj(self.vit(pixel_values=images).last_hidden_state.mean(dim=1))

        clip_text = self.clip.get_text_features(input_ids=text_ids, attention_mask=text_mask)
        clip_image = self.clip.get_image_features(pixel_values=images)
        clip_feat = self.clip_proj(clip_text * clip_image)  # 元素积融合

        topic_feat, _ = self.topic_vae(bow_vector)

        # 简单的拼接融合
        mns_features = text_feat + image_feat + clip_feat + topic_feat
        return mns_features, text_feat, image_feat

    def forward(self, text_ids, text_mask, images, bow_vector, graph_batch):
        mns_features, text_feat, image_feat = self.get_semantic_features(text_ids, text_mask, images, bow_vector)
        graph_features = self.graph_transformer(graph_batch)

        # 计算不一致性分数
        inconsistency_img_text = 1 - F.cosine_similarity(image_feat, text_feat, dim=1)
        inconsistency_graph_sem = 1 - F.cosine_similarity(graph_features, mns_features, dim=1)

        # 拼接所有特征进行分类
        final_features = torch.cat([mns_features, inconsistency_img_text.unsqueeze(1) * mns_features,
                                    inconsistency_graph_sem.unsqueeze(1) * mns_features], dim=1)

        logits = self.classifier(final_features)
        return logits, mns_features, graph_features


# --- 数据集 ---
class SpamPropagationDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, user_map, biz_map, adj_matrix):
        self.df = df
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.user_map = user_map
        self.biz_map = biz_map
        self.adj_matrix = adj_matrix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['content'])
        label = int(row['is_recommended'])

        # 图像处理
        image_path = ""
        photo_ids = str(row.get('photo_ids', ''))
        if photo_ids and photo_ids != 'nan':
            image_path = os.path.join(CONFIG["image_dir"], f"{photo_ids.split('#')[0]}.jpg")
        try:
            image = Image.open(image_path).convert('RGB') if os.path.exists(image_path) else Image.new('RGB',
                                                                                                       (224, 224))
        except:
            image = Image.new('RGB', (224, 224))

        # BOW向量用于主题模型
        tokenized = self.tokenizer(text, truncation=True)
        bow_vector = torch.zeros(self.tokenizer.vocab_size)
        for token_id in tokenized['input_ids']:
            bow_vector[token_id] = 1

        # 模拟传播图
        user_idx = self.user_map.get(row['author_name'], -1)
        biz_idx = self.biz_map.get(row['biz_name'], -1)

        node_indices = []
        if user_idx != -1: node_indices.append(user_idx)
        if biz_idx != -1: node_indices.append(biz_idx + len(self.user_map))

        # 获取一阶邻居
        neighbors = set(node_indices)
        for node in node_indices:
            neighbors.update(self.adj_matrix[node].nonzero()[0])

        subgraph_nodes = sorted(list(neighbors))
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}

        edge_list = []
        for i in subgraph_nodes:
            for j in self.adj_matrix[i].nonzero()[0]:
                if j in subgraph_nodes:
                    edge_list.append([node_map[i], node_map[j]])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0),
                                                                                                              dtype=torch.long)

        # 节点特征（简化为随机）
        node_features = torch.randn(len(subgraph_nodes), self.tokenizer.vocab_size)

        graph_data = Data(x=node_features, edge_index=edge_index)

        return {
            'text': text, 'image': image, 'label': torch.tensor(label, dtype=torch.long),
            'bow_vector': bow_vector, 'graph_data': graph_data
        }


# --- 主执行块 ---
if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content', 'author_name', 'biz_name'], inplace=True)

    # 构建全局图用于模拟
    users = df['author_name'].unique()
    businesses = df['biz_name'].unique()
    user_map = {name: i for i, name in enumerate(users)}
    biz_map = {name: i for i, name in enumerate(businesses)}
    num_nodes = len(users) + len(businesses)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for _, row in df.iterrows():
        u_idx = user_map[row['author_name']]
        b_idx = biz_map[row['biz_name']] + len(users)
        adj_matrix[u_idx, b_idx] = 1
        adj_matrix[b_idx, u_idx] = 1

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    image_processor = ViTImageProcessor.from_pretrained(CONFIG["vit_model"])

    train_dataset = SpamPropagationDataset(train_df, tokenizer, image_processor, user_map, biz_map, adj_matrix)
    val_dataset = SpamPropagationDataset(val_df, tokenizer, image_processor, user_map, biz_map, adj_matrix)
    test_dataset = SpamPropagationDataset(test_df, tokenizer, image_processor, user_map, biz_map, adj_matrix)


    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        images = [item['image'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        bow_vectors = torch.stack([item['bow_vector'] for item in batch])
        graph_data_list = [item['graph_data'] for item in batch]

        text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        image_inputs = image_processor(images, return_tensors='pt')
        graph_batch = Batch.from_data_list(graph_data_list)

        return {
            'text_ids': text_inputs['input_ids'], 'text_mask': text_inputs['attention_mask'],
            'images': image_inputs['pixel_values'], 'labels': labels,
            'bow_vector': bow_vectors, 'graph_batch': graph_batch
        }


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = CRFND(CONFIG, vocab_size=tokenizer.vocab_size).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            optimizer.zero_grad()
            logits, mns_feat, graph_feat = model(
                text_ids=batch['text_ids'].to(CONFIG["device"]), text_mask=batch['text_mask'].to(CONFIG["device"]),
                images=batch['images'].to(CONFIG["device"]), bow_vector=batch['bow_vector'].to(CONFIG["device"]),
                graph_batch=batch['graph_batch'].to(CONFIG["device"])
            )

            # 对比学习损失 (简化版)
            labels = batch['labels'].to(CONFIG["device"])
            positive_mask = (labels == 1)
            if positive_mask.sum() > 1:
                pos_mns = mns_feat[positive_mask]
                pos_graph = graph_feat[positive_mask]
                sim_matrix = F.cosine_similarity(pos_mns.unsqueeze(1), pos_graph.unsqueeze(0), dim=2)
                loss_cl = F.cross_entropy(sim_matrix, torch.arange(len(pos_mns)).to(CONFIG["device"]))
            else:
                loss_cl = 0.0

            loss_ce = F.cross_entropy(logits, labels)
            loss = loss_ce + 0.1 * loss_cl  # 组合损失
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                logits, _, _ = model(
                    text_ids=batch['text_ids'].to(CONFIG["device"]), text_mask=batch['text_mask'].to(CONFIG["device"]),
                    images=batch['images'].to(CONFIG["device"]), bow_vector=batch['bow_vector'].to(CONFIG["device"]),
                    graph_batch=batch['graph_batch'].to(CONFIG["device"])
                )
                val_correct += (logits.argmax(1) == batch['labels'].to(CONFIG["device"])).sum().item()
        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch + 1}: Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])

    # 测试
    print("\nTesting...")
    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, _, _ = model(
                text_ids=batch['text_ids'].to(CONFIG["device"]), text_mask=batch['text_mask'].to(CONFIG["device"]),
                images=batch['images'].to(CONFIG["device"]), bow_vector=batch['bow_vector'].to(CONFIG["device"]),
                graph_batch=batch['graph_batch'].to(CONFIG["device"])
            )
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)
    print(f"Test Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")