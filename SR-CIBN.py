import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor
from torch.optim import AdamW
from PIL import Image
import pandas as pd
import argparse
import numpy as np
from scipy.fft import dct
from tqdm import tqdm
import os
import warnings
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Suppress warnings from transformers and other libraries
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train SR-CIBN Model')
parser.add_argument('--batch_size', '-bs', type=int, default=16, help='Batch size for training')
parser.add_argument('--num_epochs', '-ne', type=int, default=10, help='Number of epochs for training')
args = parser.parse_args()
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs

# --- 1. Configuration ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "image_dir": "../spams_detection/spam_datasets/crawler/LA/images/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": NUM_EPOCHS,  # The paper uses 120, but 20 is a good start for testing
    "batch_size": BATCH_SIZE,  # The paper uses 128, adjust based on your GPU VRAM
    "learning_rate": 1e-4,  # Paper uses 1e-3, but 1e-4 is often safer for transformers
    "embedding_dim": 256,  # d in the paper
    "attention_heads": 8,  # N_h in the paper
    "sparsity_rate": 0.6,  # p in the paper
    "triplet_margin": 0.5,  # γ in the paper
    "matching_threshold": 0.4,  # τ in the paper
    "lambda1_con": 0.3,  # λ1 for contrastive loss
    "lambda2_triplet": 0.5,  # λ2 for triplet loss
    "contrastive_temp": 0.05,  # τ for contrastive loss
    "bert_model": "bert-base-uncased",
    "vit_model": "google/vit-base-patch16-224-in21k",
    "best_model_path": "sr_cibn_best_model.pth"
}

print(f"Using device: {CONFIG['device']}")


# --- 2. Helper Modules (Unchanged) ---
class DCTLayer(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, x):
        x_np = x.cpu().numpy()
        dct_x = dct(dct(x_np, axis=2, norm='ortho'), axis=3, norm='ortho')
        return torch.from_numpy(dct_x).to(x.device).float()


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.attention(query, key_value, key_value)
        return self.layer_norm(query + attn_output)


class GatedMultimodalUnit(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, embed_dim)
        self.w2 = nn.Linear(embed_dim, embed_dim)
        self.w3 = nn.Linear(2 * embed_dim, 1)

    def forward(self, x1, x2):
        h1, h2 = torch.tanh(self.w1(x1)), torch.tanh(self.w2(x2))
        z = torch.sigmoid(self.w3(torch.cat([x1, x2], dim=-1)))
        return z * h1 + (1 - z) * h2


# --- 3. The Main SR-CIBN Model (Unchanged) ---
class SR_CIBN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config["embedding_dim"]
        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.text_proj = nn.Linear(self.bert.config.hidden_size, self.embed_dim)
        self.vit = ViTModel.from_pretrained(config["vit_model"])
        self.vit_proj = nn.Linear(self.vit.config.hidden_size, self.embed_dim)
        self.dct = DCTLayer()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        self.freq_proj = nn.Linear(2048, self.embed_dim)
        self.cma1 = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.cma2 = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.cma3 = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.gmu1 = GatedMultimodalUnit(self.embed_dim)
        self.gmu2 = GatedMultimodalUnit(self.embed_dim)
        self.patch_enhancer = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, self.embed_dim),
                                            nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.consistency_attention = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.inconsistency_attention = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.classifier = nn.Linear(2 * self.embed_dim, 2)

    def _extract_unimodal_features(self, text_input, image_input):
        text_outputs = self.bert(**text_input)
        u_t = self.text_proj(text_outputs.last_hidden_state[:, 0, :])
        vit_outputs = self.vit(image_input)
        image_patches = vit_outputs.last_hidden_state[:, 1:, :]
        image_global = vit_outputs.last_hidden_state[:, 0, :]
        u_v_patches = self.vit_proj(image_patches)
        u_v_global = self.vit_proj(image_global)
        inception_input = F.interpolate(image_input, size=(299, 299), mode='bilinear', align_corners=False)
        dct_image = self.dct(inception_input)
        if dct_image.shape[1] == 1: dct_image = dct_image.repeat(1, 3, 1, 1)
        inception_output = self.inception(dct_image)
        freq_feat = inception_output.logits if self.training and isinstance(inception_output,
                                                                            tuple) else inception_output
        u_f = self.freq_proj(freq_feat)
        return u_t, u_v_global, u_f, u_v_patches

    def _calculate_contrastive_loss(self, u_t, u_v, u_f):
        u_t, u_v, u_f = F.normalize(u_t), F.normalize(u_v), F.normalize(u_f)
        batch_size = u_t.shape[0]
        if batch_size <= 1: return torch.tensor(0.0, device=self.config["device"])
        labels = torch.arange(batch_size, device=self.config["device"])
        sim_t2v = u_t @ u_v.T / self.config["contrastive_temp"]
        sim_v2t = u_v @ u_t.T / self.config["contrastive_temp"]
        sim_v2f = u_v @ u_f.T / self.config["contrastive_temp"]
        sim_f2v = u_f @ u_v.T / self.config["contrastive_temp"]
        return (F.cross_entropy(sim_t2v, labels) + F.cross_entropy(sim_v2t, labels) + F.cross_entropy(sim_v2f,
                                                                                                      labels) + F.cross_entropy(
            sim_f2v, labels)) / 4.0

    def _extract_consistency_inconsistency(self, u_t, u_v_patches, O_tvf):
        u_t_expanded = u_t.unsqueeze(1).expand_as(u_v_patches)
        cross_scores = torch.sum(u_v_patches * u_t_expanded, dim=-1)
        self_scores = torch.sum(u_v_patches * u_v_patches.mean(dim=1, keepdim=True), dim=-1)
        a_M = F.softmax(cross_scores, dim=-1) + F.softmax(self_scores, dim=-1)
        a_I = F.softmax(-cross_scores, dim=-1) + F.softmax(self_scores, dim=-1)
        num_select = int(self.config["sparsity_rate"] * u_v_patches.shape[1])
        _, top_M_indices = torch.topk(a_M, k=num_select, dim=1)
        _, top_I_indices = torch.topk(a_I, k=num_select, dim=1)
        V_M = torch.gather(u_v_patches, 1, top_M_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        V_I = torch.gather(u_v_patches, 1, top_I_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        V_M_enhanced, V_I_enhanced = V_M + self.patch_enhancer(V_M), V_I + self.patch_enhancer(V_I)
        O_tvf_expanded = O_tvf.unsqueeze(1)
        M_all = self.consistency_attention(V_M_enhanced, O_tvf_expanded).mean(dim=1)
        I_all = self.inconsistency_attention(V_I_enhanced, O_tvf_expanded).mean(dim=1)
        G_M, G_I = torch.cat([M_all, O_tvf], dim=-1), torch.cat([I_all, O_tvf], dim=-1)
        return G_M, G_I

    def forward(self, text_input, image_input, labels=None):
        u_t, u_v_global, u_f, u_v_patches = self._extract_unimodal_features(text_input, image_input)
        l_con = self._calculate_contrastive_loss(u_t, u_v_global, u_f)
        u_t_seq, u_v_seq, u_f_seq = u_t.unsqueeze(1), u_v_global.unsqueeze(1), u_f.unsqueeze(1)
        F1 = self.cma1(u_v_seq, u_t_seq)
        F2 = self.cma2(F1, u_t_seq)
        F3 = self.cma3(F2, u_f_seq)
        H = self.gmu1(F3.squeeze(1), F2.squeeze(1))
        O_tvf = self.gmu2(H, F1.squeeze(1))
        G_M, G_I = self._extract_consistency_inconsistency(u_t, u_v_patches, O_tvf)
        sim = F.cosine_similarity(u_t, u_v_global, dim=-1)
        p_ij = torch.sigmoid(sim)
        a = (p_ij - p_ij.min()) / (p_ij.max() - p_ij.min() + 1e-8)
        a = a.unsqueeze(-1)
        final_feature = a * G_I + (1 - a) * G_M
        logits = self.classifier(final_feature)
        l_triplet = torch.tensor(0.0, device=self.config["device"])
        if self.training and labels is not None and len(labels) > 1:
            consistent_mask = (a.squeeze() >= self.config["matching_threshold"])
            if consistent_mask.sum() > 1: l_triplet += self._calculate_triplet_loss_for_cluster(
                final_feature[consistent_mask], labels[consistent_mask])
            if (~consistent_mask).sum() > 1: l_triplet += self._calculate_triplet_loss_for_cluster(
                final_feature[~consistent_mask], labels[~consistent_mask])
        return logits, l_con, l_triplet

    def _calculate_triplet_loss_for_cluster(self, features, labels):
        loss, count = 0.0, 0
        dist_matrix = torch.cdist(features, features, p=2)
        for i in range(len(labels)):
            anchor_label = labels[i]
            pos_mask, neg_mask = (labels == anchor_label), (labels != anchor_label)
            pos_mask[i] = False
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                hardest_pos = dist_matrix[i][pos_mask].max()
                hardest_neg = dist_matrix[i][neg_mask].min()
                loss += F.relu(hardest_pos - hardest_neg + self.config["triplet_margin"])
                count += 1
        return loss / (count + 1e-8)


# --- 4. Custom PyTorch Dataset (Unchanged) ---
class SpamReviewDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, config):
        self.df, self.tokenizer, self.image_processor, self.config = df, tokenizer, image_processor, config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['content'])
        label = int(row['is_recommended'])
        text_encoding = self.tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        image_path = ""
        photo_ids = str(row.get('photo_ids', ''))
        if photo_ids and photo_ids != 'nan':
            first_photo_id = photo_ids.split('#')[0]
            image_path = os.path.join(self.config["image_dir"], f"{first_photo_id}.jpg")
        try:
            image = Image.open(image_path).convert('RGB') if os.path.exists(image_path) else Image.new('RGB',
                                                                                                       (224, 224))
        except Exception:
            image = Image.new('RGB', (224, 224))
        processed_image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return {'text_input_ids': text_encoding['input_ids'].flatten(),
                'text_attention_mask': text_encoding['attention_mask'].flatten(), 'image': processed_image,
                'label': torch.tensor(label, dtype=torch.long)}


# --- 5. Training and Evaluation Functions ---
def train_epoch(model, data_loader, optimizer, device, config):
    model.train()
    total_loss, total_cls_loss, total_con_loss, total_tri_loss, total_correct = 0, 0, 0, 0, 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        text_input = {'input_ids': batch['text_input_ids'].to(device),
                      'attention_mask': batch['text_attention_mask'].to(device)}
        images, labels = batch['image'].to(device), batch['label'].to(device)
        logits, l_con, l_triplet = model(text_input, images, labels)
        l_cls = F.cross_entropy(logits, labels)
        loss = l_cls + config["lambda1_con"] * l_con + config["lambda2_triplet"] * l_triplet
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_cls_loss += l_cls.item()
        total_con_loss += l_con.item()
        total_tri_loss += l_triplet.item()
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
    n = len(data_loader)
    return total_correct / len(
        data_loader.dataset), total_loss / n, total_cls_loss / n, total_con_loss / n, total_tri_loss / n


def eval_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            text_input = {'input_ids': batch['text_input_ids'].to(device),
                          'attention_mask': batch['text_attention_mask'].to(device)}
            images, labels = batch['image'].to(device), batch['label'].to(device)
            logits, _, _ = model(text_input, images)
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
    return total_correct / len(data_loader.dataset)


# --- NEW: Test function with detailed metrics ---
def test_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            text_input = {'input_ids': batch['text_input_ids'].to(device),
                          'attention_mask': batch['text_attention_mask'].to(device)}
            images, labels = batch['image'].to(device), batch['label'].to(device)
            logits, _, _ = model(text_input, images)
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    # Use weighted average for precision, recall, f1 to account for class imbalance
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)

    print("\n--- Test Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("--------------------")


# --- 6. Main Execution Block ---
if __name__ == '__main__':
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content'], inplace=True)

    # --- MODIFIED: 80/10/10 split for train/val/test ---
    train_df, val_df, test_df = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    image_processor = ViTImageProcessor.from_pretrained(CONFIG["vit_model"])

    train_dataset = SpamReviewDataset(train_df, tokenizer, image_processor, CONFIG)
    val_dataset = SpamReviewDataset(val_df, tokenizer, image_processor, CONFIG)
    test_dataset = SpamReviewDataset(test_df, tokenizer, image_processor, CONFIG)  # --- NEW ---

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)  # --- NEW ---

    model = SR_CIBN(CONFIG).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    best_val_acc = 0.0  # --- NEW ---

    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        train_acc, train_loss, cls_loss, con_loss, tri_loss = train_epoch(model, train_loader, optimizer,
                                                                          CONFIG["device"], CONFIG)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  -> CLS Loss: {cls_loss:.4f}, CON Loss: {con_loss:.4f}, TRI Loss: {tri_loss:.4f}")

        val_acc = eval_model(model, val_loader, CONFIG["device"])
        print(f"Val Acc: {val_acc:.4f}")

        # --- NEW: Save the best model based on validation accuracy ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    print("\nTraining complete.")

    # --- NEW: Final testing phase ---
    print("\nLoading best model for final testing...")
    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    test_model(model, test_loader, CONFIG["device"])