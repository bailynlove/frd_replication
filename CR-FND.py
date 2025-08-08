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
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from collections import defaultdict

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train CR-FND Model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
args = parser.parse_args()

CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "image_dir": "../spams_detection/spam_datasets/crawler/LA/images/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 3,
    "learning_rate": 1e-5,
    "bert_model": "bert-base-uncased",
    "vit_model": "google/vit-base-patch16-224-in21k",
    "clip_model": "openai/clip-vit-base-patch32",
    "hidden_dim": 256,
    "num_classes": 2,
    "best_model_path": "cr_fnd_best_model_v3.pth",
    "grad_clip_value": 1.0
}

print(f"使用设备: {CONFIG['device']}")


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class GraphTransformer(nn.Module):
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


class CRFND(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.vit = ViTModel.from_pretrained(config["vit_model"])
        self.clip = CLIPModel.from_pretrained(config["clip_model"])

        bert_dim, vit_dim, clip_dim = self.bert.config.hidden_size, self.vit.config.hidden_size, self.clip.config.text_config.hidden_size
        self.topic_vae = VAE(vocab_size, 512, config["hidden_dim"])

        self.text_proj = nn.Sequential(nn.Linear(bert_dim, config["hidden_dim"]), nn.LayerNorm(config["hidden_dim"]))
        self.image_proj = nn.Sequential(nn.Linear(vit_dim, config["hidden_dim"]), nn.LayerNorm(config["hidden_dim"]))
        self.clip_proj = nn.Sequential(nn.Linear(clip_dim, config["hidden_dim"]), nn.LayerNorm(config["hidden_dim"]))
        self.topic_proj = nn.Sequential(nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                                        nn.LayerNorm(config["hidden_dim"]))

        self.graph_transformer = GraphTransformer(bert_dim, config["hidden_dim"])

        self.mns_fusion = nn.Sequential(
            nn.Linear(config["hidden_dim"] * 4, config["hidden_dim"]),
            nn.ReLU(),
            nn.LayerNorm(config["hidden_dim"])
        )

        self.classifier = nn.Sequential(
            nn.Linear(config["hidden_dim"] * 3, config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config["hidden_dim"], config["num_classes"])
        )

    def get_semantic_features(self, bert_input_ids, bert_attention_mask, clip_input_ids, clip_attention_mask, images,
                              bow_vector):
        text_feat = self.text_proj(
            self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask).last_hidden_state.mean(dim=1))
        image_feat = self.image_proj(self.vit(pixel_values=images).last_hidden_state.mean(dim=1))

        with torch.no_grad():
            clip_text = self.clip.get_text_features(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
            clip_image = self.clip.get_image_features(pixel_values=images)
        clip_feat = self.clip_proj(clip_text * clip_image)

        topic_feat_z, _, _ = self.topic_vae(bow_vector)
        topic_feat = self.topic_proj(topic_feat_z)

        mns_features = self.mns_fusion(torch.cat([text_feat, image_feat, clip_feat, topic_feat], dim=1))
        return mns_features, text_feat, image_feat

    def forward(self, bert_input_ids, bert_attention_mask, clip_input_ids, clip_attention_mask, images, bow_vector,
                graph_batch):
        mns_features, text_feat, image_feat = self.get_semantic_features(bert_input_ids, bert_attention_mask,
                                                                         clip_input_ids, clip_attention_mask, images,
                                                                         bow_vector)
        graph_features = self.graph_transformer(graph_batch)

        epsilon = 1e-8
        inconsistency_img_text = 1 - F.cosine_similarity(image_feat, text_feat, dim=1, eps=epsilon)
        inconsistency_graph_sem = 1 - F.cosine_similarity(graph_features, mns_features, dim=1, eps=epsilon)

        final_features = torch.cat([mns_features, inconsistency_img_text.unsqueeze(1) * mns_features,
                                    inconsistency_graph_sem.unsqueeze(1) * mns_features], dim=1)

        logits = self.classifier(final_features)
        return logits, mns_features, graph_features


class SpamPropagationDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, user_map, biz_map, adj_matrix):
        self.df, self.tokenizer, self.image_processor = df, tokenizer, image_processor
        self.user_map, self.biz_map, self.adj_matrix = user_map, biz_map, adj_matrix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['content'])
        label = int(row['is_recommended'])

        image_path = ""
        photo_ids = str(row.get('photo_ids', ''))
        if photo_ids and photo_ids != 'nan':
            image_path = os.path.join(CONFIG["image_dir"], f"{photo_ids.split('#')[0]}.jpg")
        try:
            image = Image.open(image_path).convert('RGB') if os.path.exists(image_path) else Image.new('RGB',
                                                                                                       (224, 224))
        except:
            image = Image.new('RGB', (224, 224))

        tokenized = self.tokenizer(text, truncation=True, max_length=512)
        bow_vector = torch.zeros(self.tokenizer.vocab_size)
        for token_id in tokenized['input_ids']:
            bow_vector[token_id] += 1

        user_idx, biz_idx = self.user_map.get(row['author_name'], -1), self.biz_map.get(row['biz_name'], -1)

        node_indices = []
        if user_idx != -1: node_indices.append(user_idx)
        if biz_idx != -1: node_indices.append(biz_idx + len(self.user_map))

        neighbors = set(node_indices)
        for node in node_indices:
            neighbors.update(self.adj_matrix[node].nonzero()[0])

        subgraph_nodes = sorted(list(neighbors))
        if not subgraph_nodes: subgraph_nodes = [0]
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}

        edge_list = []
        for i in subgraph_nodes:
            for j in self.adj_matrix[i].nonzero()[0]:
                if j in subgraph_nodes:
                    edge_list.append([node_map[i], node_map[j]])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0),
                                                                                                              dtype=torch.long)

        # --- 修正点: 将节点输入作为独立属性 ---
        node_texts = [f"user: {name}" for name, idx in self.user_map.items() if idx in subgraph_nodes] + \
                     [f"business: {name}" for name, idx in self.biz_map.items() if
                      idx + len(self.user_map) in subgraph_nodes]
        if not node_texts: node_texts = ["<pad>"]

        node_inputs = self.tokenizer(node_texts, padding='max_length', max_length=32, truncation=True,
                                     return_tensors='pt')

        graph_data = Data(
            edge_index=edge_index,
            input_ids=node_inputs['input_ids'],
            attention_mask=node_inputs['attention_mask']
        )

        return {
            'text': text, 'image': image, 'label': torch.tensor(label, dtype=torch.long),
            'bow_vector': bow_vector, 'graph_data': graph_data
        }


if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content', 'author_name', 'biz_name'], inplace=True)

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
    image_processor = ViTImageProcessor.from_pretrained(CONFIG["vit_model"])

    train_dataset = SpamPropagationDataset(train_df, tokenizer, image_processor, user_map, biz_map, adj_matrix)
    val_dataset = SpamPropagationDataset(val_df, tokenizer, image_processor, user_map, biz_map, adj_matrix)
    test_dataset = SpamPropagationDataset(test_df, tokenizer, image_processor, user_map, biz_map, adj_matrix)


    def collate_fn(batch):
        texts, images, labels = [item['text'] for item in batch], [item['image'] for item in batch], torch.stack(
            [item['label'] for item in batch])
        bow_vectors, graph_data_list = torch.stack([item['bow_vector'] for item in batch]), [item['graph_data'] for item
                                                                                             in batch]
        bert_inputs = tokenizer(texts, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        clip_inputs = tokenizer(texts, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        image_inputs = image_processor(images, return_tensors='pt')
        graph_batch = Batch.from_data_list(graph_data_list)
        return {
            'bert_input_ids': bert_inputs['input_ids'], 'bert_attention_mask': bert_inputs['attention_mask'],
            'clip_input_ids': clip_inputs['input_ids'], 'clip_attention_mask': clip_inputs['attention_mask'],
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
        total_loss, total_correct = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            optimizer.zero_grad()

            # --- 修正点: 正确处理 graph_batch ---
            graph_batch = batch['graph_batch'].to(CONFIG['device'])
            with torch.no_grad():
                node_features = model.bert(
                    input_ids=graph_batch.input_ids,
                    attention_mask=graph_batch.attention_mask
                ).last_hidden_state.mean(dim=1)
            graph_batch.x = node_features

            logits, mns_feat, graph_feat = model(
                bert_input_ids=batch['bert_input_ids'].to(CONFIG["device"]),
                bert_attention_mask=batch['bert_attention_mask'].to(CONFIG["device"]),
                clip_input_ids=batch['clip_input_ids'].to(CONFIG["device"]),
                clip_attention_mask=batch['clip_attention_mask'].to(CONFIG["device"]),
                images=batch['images'].to(CONFIG["device"]),
                bow_vector=batch['bow_vector'].to(CONFIG["device"]),
                graph_batch=graph_batch
            )

            labels = batch['labels'].to(CONFIG["device"])
            positive_mask = (labels == 1)
            loss_cl = 0.0
            if positive_mask.sum() > 1:
                pos_mns, pos_graph = mns_feat[positive_mask], graph_feat[positive_mask]
                sim_matrix = F.cosine_similarity(pos_mns.unsqueeze(1), pos_graph.unsqueeze(0), dim=2)
                loss_cl = F.cross_entropy(sim_matrix, torch.arange(len(pos_mns), device=CONFIG["device"]))

            loss_ce = F.cross_entropy(logits, labels)
            loss = loss_ce + 0.1 * loss_cl

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_value"])
                optimizer.step()
                total_loss += loss.item()

            total_correct += (logits.argmax(1) == labels).sum().item()

        train_acc = total_correct / len(train_dataset)
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                graph_batch = batch['graph_batch'].to(CONFIG['device'])
                with torch.no_grad():
                    node_features = model.bert(
                        input_ids=graph_batch.input_ids,
                        attention_mask=graph_batch.attention_mask
                    ).last_hidden_state.mean(dim=1)
                graph_batch.x = node_features

                logits, _, _ = model(
                    bert_input_ids=batch['bert_input_ids'].to(CONFIG["device"]),
                    bert_attention_mask=batch['bert_attention_mask'].to(CONFIG["device"]),
                    clip_input_ids=batch['clip_input_ids'].to(CONFIG["device"]),
                    clip_attention_mask=batch['clip_attention_mask'].to(CONFIG["device"]),
                    images=batch['images'].to(CONFIG["device"]),
                    bow_vector=batch['bow_vector'].to(CONFIG["device"]),
                    graph_batch=graph_batch
                )
                val_correct += (logits.argmax(1) == batch['labels'].to(CONFIG["device"])).sum().item()
        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])

    print("\nTesting...")
    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            graph_batch = batch['graph_batch'].to(CONFIG['device'])
            with torch.no_grad():
                node_features = model.bert(
                    input_ids=graph_batch.input_ids,
                    attention_mask=graph_batch.attention_mask
                ).last_hidden_state.mean(dim=1)
            graph_batch.x = node_features

            logits, _, _ = model(
                bert_input_ids=batch['bert_input_ids'].to(CONFIG["device"]),
                bert_attention_mask=batch['bert_attention_mask'].to(CONFIG["device"]),
                clip_input_ids=batch['clip_input_ids'].to(CONFIG["device"]),
                clip_attention_mask=batch['clip_attention_mask'].to(CONFIG["device"]),
                images=batch['images'].to(CONFIG["device"]),
                bow_vector=batch['bow_vector'].to(CONFIG["device"]),
                graph_batch=graph_batch
            )
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)
    print(f"Test Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")