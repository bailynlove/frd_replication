import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor, CLIPModel, CLIPProcessor
from torch.optim import AdamW
from PIL import Image
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os
import warnings
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import spacy

# Suppress warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train SR-CIBN Model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
args = parser.parse_args()
BATCH_SIZE = args.batch_size

# --- 1. Configuration ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "image_dir": "../spams_detection/spam_datasets/crawler/LA/images/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 10,
    "batch_size": BATCH_SIZE,  # Reduced batch size due to high memory usage of GATs and multiple models
    "learning_rate": 1e-4,
    "embedding_dim": 768,
    "attention_heads": 4,
    "gat_out_features": 128,
    "bert_model": "bert-base-uncased",
    "vit_model": "google/vit-base-patch16-224-in21k",
    "clip_model": "openai/clip-vit-base-patch32",
    "spacy_model": "en_core_web_sm",
    "best_model_path": "mgc_net_best_model.pth"
}

print(f"Using device: {CONFIG['device']}")


# --- 2. Helper Modules ---

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, alpha=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.a = nn.Parameter(torch.randn(size=(1, n_heads, 2 * out_features)))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, adj):
        N = h.size(0)
        if N == 0:
            return torch.zeros(0, self.W.out_features, device=h.device)

        Wh = self.W(h).view(N, self.n_heads, -1)
        out_features = Wh.size(-1)

        Wh_i = Wh.unsqueeze(1).expand(N, N, self.n_heads, out_features)
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.n_heads, out_features)
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)

        e = self.leakyrelu((a_input * self.a).sum(dim=-1))

        attention = torch.where(adj.unsqueeze(-1) > 0, e, -9e15 * torch.ones_like(e))
        attention = self.softmax(attention)

        h_prime = torch.einsum('ijh,jhd->ihd', attention, Wh)
        return F.elu(h_prime.reshape(N, self.n_heads * out_features))


# --- 3. The Main Model ---
class MGC_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.embed_dim = config["embedding_dim"]

        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
        self.vit = ViTModel.from_pretrained(config["vit_model"])

        self.token_cross_attention = nn.MultiheadAttention(self.embed_dim, 8, batch_first=True)
        self.nlp = spacy.load(config["spacy_model"])

        gat_out_dim = config["gat_out_features"] * config["attention_heads"]
        self.text_gat = GATLayer(self.embed_dim, config["gat_out_features"], config["attention_heads"])
        self.image_gat = GATLayer(self.embed_dim, config["gat_out_features"], config["attention_heads"])

        self.clip = CLIPModel.from_pretrained(config["clip_model"])
        self.clip_processor = CLIPProcessor.from_pretrained(config["clip_model"])

        self.fusion_mlp = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 3), nn.Sigmoid())

        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.phrase_proj = nn.Linear(gat_out_dim, self.embed_dim)
        self.global_proj = nn.Linear(self.clip.config.text_config.hidden_size, self.embed_dim)

        self.classifier = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 2), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(self.embed_dim // 2, 2))

    def _get_text_graph(self, text_sample):
        doc = self.nlp(text_sample)
        num_tokens = len(doc)
        adj = np.zeros((num_tokens, num_tokens))
        for token in doc:
            if token.i < num_tokens:
                adj[token.i, token.i] = 1
                for child in token.children:
                    if child.i < num_tokens:
                        adj[token.i, child.i] = 1
                        adj[child.i, token.i] = 1
        return torch.from_numpy(adj).float().to(self.device)

    def _get_image_graph(self, num_patches=196):
        grid_size = int(np.sqrt(num_patches))
        adj = np.zeros((num_patches, num_patches))
        for i in range(num_patches):
            adj[i, i] = 1
            row, col = i // grid_size, i % grid_size
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        adj[i, nr * grid_size + nc] = 1
        return torch.from_numpy(adj).float().to(self.device)

    def forward(self, text_raw, image_input):
        text_inputs = self.tokenizer(text_raw, padding='max_length', max_length=64, truncation=True,
                                     return_tensors="pt").to(self.device)
        text_features = self.bert(**text_inputs).last_hidden_state
        image_features = self.vit(image_input).last_hidden_state[:, 1:, :]

        updated_text_features, _ = self.token_cross_attention(text_features, image_features, image_features)
        c_token_matrix = torch.bmm(updated_text_features, image_features.transpose(1, 2))
        s_token = torch.diagonal(c_token_matrix, dim1=-2, dim2=-1).mean(dim=1)
        f_token = self.token_proj(updated_text_features.mean(dim=1))

        batch_size = text_features.shape[0]
        bert_seq_len = text_features.shape[1]
        updated_text_gat_list, updated_image_gat_list = [], []
        image_adj = self._get_image_graph(image_features.shape[1])

        for i in range(batch_size):
            # --- START OF FIX ---
            text_adj_spacy = self._get_text_graph(text_raw[i])
            spacy_len = text_adj_spacy.shape[0]
            text_adj = torch.zeros((bert_seq_len, bert_seq_len), device=self.device)
            copy_len = min(spacy_len, bert_seq_len)
            text_adj[:copy_len, :copy_len] = text_adj_spacy[:copy_len, :copy_len]
            text_feat_i = text_features[i]
            # --- END OF FIX ---

            text_gat_out = self.text_gat(text_feat_i, text_adj)
            if text_gat_out.size(0) > 0:
                updated_text_gat_list.append(text_gat_out.mean(dim=0))
            else:
                updated_text_gat_list.append(torch.zeros(self.text_gat.W.out_features, device=self.device))

            image_gat_out = self.image_gat(image_features[i], image_adj)
            updated_image_gat_list.append(image_gat_out.mean(dim=0))

        text_gat_features = torch.stack(updated_text_gat_list)
        image_gat_features = torch.stack(updated_image_gat_list)
        s_phrase = F.cosine_similarity(text_gat_features, image_gat_features, dim=1)
        f_phrase = self.phrase_proj(text_gat_features)

        clip_inputs = self.clip_processor(text=text_raw, images=image_input, return_tensors="pt", padding=True).to(
            self.device)
        clip_outputs = self.clip(**clip_inputs)
        s_global = F.cosine_similarity(clip_outputs.text_embeds, clip_outputs.image_embeds, dim=1)
        f_global = self.global_proj(clip_outputs.text_embeds)

        consistency_scores = torch.stack([s_token, s_phrase, s_global], dim=1).detach()
        fusion_weights = self.fusion_mlp(consistency_scores)
        w_token, w_phrase, w_global = fusion_weights[:, 0], fusion_weights[:, 1], fusion_weights[:, 2]

        f_agg = (w_token.unsqueeze(1) * f_token +
                 w_phrase.unsqueeze(1) * f_phrase +
                 w_global.unsqueeze(1) * f_global)

        return self.classifier(f_agg)


# --- 4. Dataset Class (Unchanged) ---
class SpamReviewDataset(Dataset):
    def __init__(self, df, image_processor, config):
        self.df, self.image_processor, self.config = df, image_processor, config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['content'])
        label = int(row['is_recommended'])
        image_path = ""
        photo_ids = str(row.get('photo_ids', ''))
        if photo_ids and photo_ids != 'nan':
            image_path = os.path.join(self.config["image_dir"], f"{photo_ids.split('#')[0]}.jpg")
        try:
            image = Image.open(image_path).convert('RGB') if os.path.exists(image_path) else Image.new('RGB',
                                                                                                       (224, 224))
        except Exception:
            image = Image.new('RGB', (224, 224))
        processed_image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return {'text_raw': text, 'image': processed_image, 'label': torch.tensor(label, dtype=torch.long)}


# --- 5. Training, Evaluation, and Testing Functions (Unchanged) ---
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        text_raw, images, labels = batch['text_raw'], batch['image'].to(device), batch['label'].to(device)
        logits = model(text_raw, images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
    return total_correct / len(data_loader.dataset), total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            text_raw, images, labels = batch['text_raw'], batch['image'].to(device), batch['label'].to(device)
            logits = model(text_raw, images)
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
    return total_correct / len(data_loader.dataset)


def test_model(model, data_loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            text_raw, images, labels = batch['text_raw'], batch['image'].to(device), batch['label'].to(device)
            logits = model(text_raw, images)
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)
    print("\n--- Test Results ---")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("--------------------")


# --- 6. Main Execution Block (Unchanged) ---
if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content'], inplace=True)
    train_df, val_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    image_processor = ViTImageProcessor.from_pretrained(CONFIG["vit_model"])
    train_dataset = SpamReviewDataset(train_df, image_processor, CONFIG)
    val_dataset = SpamReviewDataset(val_df, image_processor, CONFIG)
    test_dataset = SpamReviewDataset(test_df, image_processor, CONFIG)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    model = MGC_Net(CONFIG).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, CONFIG["device"])
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        val_acc = eval_model(model, val_loader, CONFIG["device"])
        print(f"Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    print("\nTraining complete. Loading best model for final testing...")
    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    test_model(model, test_loader, CONFIG["device"])