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
import numpy as np
from scipy.fft import dct
from tqdm import tqdm
import os
import warnings

# Suppress warnings from transformers and other libraries
warnings.filterwarnings("ignore")

# --- 1. Configuration ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "image_dir": "../spams_detection/spam_datasets/crawler/LA/images/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 20,  # The paper uses 120, but 20 is a good start for testing
    "batch_size": 16,  # The paper uses 128, adjust based on your GPU VRAM
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
}

print(f"Using device: {CONFIG['device']}")


# --- 2. Helper Modules ---

class DCTLayer(nn.Module):
    """Applies 2D Discrete Cosine Transform."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Apply DCT to each channel
        # Input x: (B, C, H, W)
        # We need to process this on CPU with numpy/scipy as there's no native torch DCT-II
        x_np = x.cpu().numpy()
        dct_x = dct(dct(x_np, axis=2, norm='ortho'), axis=3, norm='ortho')
        return torch.from_numpy(dct_x).to(x.device).float()


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention (CMA) module from the paper."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        # The paper uses Q from one modality and K,V from another
        attn_output, _ = self.attention(query, key_value, key_value)
        # Residual connection and LayerNorm
        output = self.layer_norm(query + attn_output)
        return output


class GatedMultimodalUnit(nn.Module):
    """Gated Multimodal Unit (GMU) from the paper (Eqs. 16-19)."""

    def __init__(self, embed_dim):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, embed_dim)
        self.w2 = nn.Linear(embed_dim, embed_dim)
        self.w3 = nn.Linear(2 * embed_dim, 1)

    def forward(self, x1, x2):
        h1 = torch.tanh(self.w1(x1))
        h2 = torch.tanh(self.w2(x2))
        z = torch.sigmoid(self.w3(torch.cat([x1, x2], dim=-1)))
        H = z * h1 + (1 - z) * h2
        return H


# --- 3. The Main SR-CIBN Model ---

class SR_CIBN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config["embedding_dim"]

        # 3.1 Unimodal Feature Extractors
        # Text Encoder
        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.text_proj = nn.Linear(self.bert.config.hidden_size, self.embed_dim)

        # Image Spatial Encoder
        self.vit = ViTModel.from_pretrained(config["vit_model"])
        self.vit_proj = nn.Linear(self.vit.config.hidden_size, self.embed_dim)

        # Image Frequency Encoder
        self.dct = DCTLayer()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        # We only need the feature extractor part
        self.inception.fc = nn.Identity()
        # InceptionV3 needs 299x299 input, we'll resize
        self.freq_proj = nn.Linear(2048, self.embed_dim)  # InceptionV3 output is 2048

        # 3.3 Hierarchical Multimodal Feature Fusion
        self.cma1 = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.cma2 = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.cma3 = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.gmu1 = GatedMultimodalUnit(self.embed_dim)
        self.gmu2 = GatedMultimodalUnit(self.embed_dim)

        # 3.4 Global Consistency/Inconsistency Features Extraction
        # This part is complex, involving patch selection and enhancement
        self.patch_enhancer = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.consistency_attention = CrossModalAttention(self.embed_dim, config["attention_heads"])
        self.inconsistency_attention = CrossModalAttention(self.embed_dim, config["attention_heads"])

        # 3.6 Final Classifier
        # The input to the classifier is from dynamic fusion, size is embed_dim
        # The paper concatenates G_I/G_M with O_tvf, so input is 2*embed_dim
        self.classifier = nn.Linear(2 * self.embed_dim, 2)  # 2 classes: real (1), fake (0)

    def _extract_consistency_inconsistency(self, u_t, u_v_patches, O_tvf):
        u_t_expanded = u_t.unsqueeze(1).expand_as(u_v_patches)

        cross_attention_scores = torch.sum(u_v_patches * u_t_expanded, dim=-1)
        self_attention_scores = torch.sum(u_v_patches * u_v_patches.mean(dim=1, keepdim=True), dim=-1)

        a_M = F.softmax(cross_attention_scores, dim=-1) + F.softmax(self_attention_scores, dim=-1)
        a_I = F.softmax(-cross_attention_scores, dim=-1) + F.softmax(self_attention_scores, dim=-1)

        num_patches = u_v_patches.shape[1]
        num_select = int(self.config["sparsity_rate"] * num_patches)

        _, top_M_indices = torch.topk(a_M, k=num_select, dim=1)
        _, top_I_indices = torch.topk(a_I, k=num_select, dim=1)

        V_M = torch.gather(u_v_patches, 1, top_M_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        V_I = torch.gather(u_v_patches, 1, top_I_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        V_M_enhanced = V_M + self.patch_enhancer(V_M)
        V_I_enhanced = V_I + self.patch_enhancer(V_I)

        O_tvf_expanded = O_tvf.unsqueeze(1)

        # --- START OF FIX ---
        # We need a single summary vector from the sequence of patches. Use .mean(dim=1).
        M_all = self.consistency_attention(V_M_enhanced, O_tvf_expanded).mean(dim=1)
        I_all = self.inconsistency_attention(V_I_enhanced, O_tvf_expanded).mean(dim=1)
        # --- END OF FIX ---

        G_M = torch.cat([M_all, O_tvf], dim=-1)
        G_I = torch.cat([I_all, O_tvf], dim=-1)

        return G_M, G_I

    def _calculate_contrastive_loss(self, u_t, u_v, u_f):
        # Normalize features for stable dot product
        u_t = F.normalize(u_t, p=2, dim=1)
        u_v = F.normalize(u_v, p=2, dim=1)
        u_f = F.normalize(u_f, p=2, dim=1)

        # Create ground truth: positive pairs are on the diagonal
        batch_size = u_t.shape[0]
        if batch_size <= 1: return torch.tensor(0.0).to(self.config["device"])

        labels = torch.arange(batch_size).to(self.config["device"])

        # t2v loss
        sim_t2v = u_t @ u_v.T / self.config["contrastive_temp"]
        loss_t2v = F.cross_entropy(sim_t2v, labels)

        # v2t loss
        sim_v2t = u_v @ u_t.T / self.config["contrastive_temp"]
        loss_v2t = F.cross_entropy(sim_v2t, labels)

        # v2f loss
        sim_v2f = u_v @ u_f.T / self.config["contrastive_temp"]
        loss_v2f = F.cross_entropy(sim_v2f, labels)

        # f2v loss
        sim_f2v = u_f @ u_v.T / self.config["contrastive_temp"]
        loss_f2v = F.cross_entropy(sim_f2v, labels)

        # Total contrastive loss
        l_con = (loss_t2v + loss_v2t + loss_v2f + loss_f2v) / 4.0
        return l_con

    def _extract_consistency_inconsistency(self, u_t, u_v_patches, O_tvf):
        # Expand text and global features to match patch dimensions
        u_t_expanded = u_t.unsqueeze(1).expand_as(u_v_patches)

        # Calculate scores (Eqs. 20-22 are simplified here with dot products)
        cross_attention_scores = torch.sum(u_v_patches * u_t_expanded, dim=-1)
        self_attention_scores = torch.sum(u_v_patches * u_v_patches.mean(dim=1, keepdim=True), dim=-1)

        # Consistency scores (Eq. 23)
        a_M = F.softmax(cross_attention_scores, dim=-1) + F.softmax(self_attention_scores, dim=-1)

        # Inconsistency scores (Eq. 24) - using the softmax(-x) trick
        a_I = F.softmax(-cross_attention_scores, dim=-1) + F.softmax(self_attention_scores, dim=-1)

        # Select top-k patches
        num_patches = u_v_patches.shape[1]
        num_select = int(self.config["sparsity_rate"] * num_patches)

        _, top_M_indices = torch.topk(a_M, k=num_select, dim=1)
        _, top_I_indices = torch.topk(a_I, k=num_select, dim=1)

        V_M = torch.gather(u_v_patches, 1, top_M_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        V_I = torch.gather(u_v_patches, 1, top_I_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Enhance patches (Eqs. 28-30)
        V_M_enhanced = V_M + self.patch_enhancer(V_M)
        V_I_enhanced = V_I + self.patch_enhancer(V_I)

        # Fuse with global features O_tvf (Eqs. 35-38)
        O_tvf_expanded = O_tvf.unsqueeze(1)
        M_all = self.consistency_attention(V_M_enhanced, O_tvf_expanded).squeeze(1)
        I_all = self.inconsistency_attention(V_I_enhanced, O_tvf_expanded).squeeze(1)

        G_M = torch.cat([M_all, O_tvf], dim=-1)
        G_I = torch.cat([I_all, O_tvf], dim=-1)

        return G_M, G_I

    def forward(self, text_input, image_input, labels=None):
        # 1. Unimodal Feature Extraction
        u_t, u_v_global, u_f, u_v_patches = self._extract_unimodal_features(text_input, image_input)

        # 2. Contrastive Loss (calculated on global features)
        l_con = self._calculate_contrastive_loss(u_t, u_v_global, u_f)

        # 3. Hierarchical Fusion
        # The paper fuses global features, so we unsqueeze to add a sequence dimension
        u_t_seq, u_v_seq, u_f_seq = u_t.unsqueeze(1), u_v_global.unsqueeze(1), u_f.unsqueeze(1)

        F1 = self.cma1(u_v_seq, u_t_seq)
        F2 = self.cma2(F1, u_t_seq)
        F3 = self.cma3(F2, u_f_seq)

        # GMU fusion needs single vectors, not sequences
        H = self.gmu1(F3.squeeze(1), F2.squeeze(1))
        O_tvf = self.gmu2(H, F1.squeeze(1))

        # 4. Consistency/Inconsistency Extraction
        G_M, G_I = self._extract_consistency_inconsistency(u_t, u_v_patches, O_tvf)

        # 5. Dynamic Fusion
        # Calculate matching degree 'a' (Eqs. 39-41)
        sim = F.cosine_similarity(u_t, u_v_global, dim=-1)
        p_ij = torch.sigmoid(sim)
        a = (p_ij - p_ij.min()) / (p_ij.max() - p_ij.min() + 1e-8)
        a = a.unsqueeze(-1)

        # Final feature F (Eq. 42)
        # The paper's logic: high 'a' (consistency) means we should check for *inconsistency*
        # So 'a' weights G_I. This is a key insight.
        # However, the paper has a typo in the text vs formula. Let's follow the logic.
        # "To detect image-text semantically consistent fake news, the model should rely more on inconsistency features"
        # "When a is high, it indicates semantic consistency"
        # So, F = a * G_I + (1-a) * G_M
        final_feature = a * G_I + (1 - a) * G_M

        # 6. Classification
        logits = self.classifier(final_feature)

        # 7. Triplet Loss
        l_triplet = torch.tensor(0.0).to(self.config["device"])
        if self.training and labels is not None and len(labels) > 1:
            # Cluster based on 'a'
            consistent_mask = (a.squeeze() >= self.config["matching_threshold"])
            inconsistent_mask = ~consistent_mask

            # Process consistent cluster
            if consistent_mask.sum() > 1:
                F_c = final_feature[consistent_mask]
                L_c = labels[consistent_mask]
                l_triplet += self._calculate_triplet_loss_for_cluster(F_c, L_c)

            # Process inconsistent cluster
            if inconsistent_mask.sum() > 1:
                F_ic = final_feature[inconsistent_mask]
                L_ic = labels[inconsistent_mask]
                l_triplet += self._calculate_triplet_loss_for_cluster(F_ic, L_ic)

        return logits, l_con, l_triplet

    def _calculate_triplet_loss_for_cluster(self, features, labels):
        loss = 0.0
        count = 0
        dist_matrix = torch.cdist(features, features, p=2)

        for i in range(len(labels)):
            anchor_label = labels[i]

            # Find positive samples (same label, not self)
            pos_mask = (labels == anchor_label)
            pos_mask[i] = False

            # Find negative samples (different label)
            neg_mask = (labels != anchor_label)

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                # Select hardest positive and hardest negative
                pos_dists = dist_matrix[i][pos_mask]
                neg_dists = dist_matrix[i][neg_mask]

                hardest_positive = pos_dists.max()
                hardest_negative = neg_dists.min()

                triplet_loss = F.relu(hardest_positive - hardest_negative + self.config["triplet_margin"])
                loss += triplet_loss
                count += 1

        return loss / (count + 1e-8)


# --- 4. Custom PyTorch Dataset ---
class SpamReviewDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, config, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config
        self.is_train = is_train

        # Define image transforms
        # ViT processor handles resize and normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row['content'])
        label = int(row['is_recommended'])

        # Text processing
        text_encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Image processing
        image_path = ""
        photo_ids = str(row.get('photo_ids', ''))
        if photo_ids and photo_ids != 'nan':
            first_photo_id = photo_ids.split('#')[0]
            image_path = os.path.join(self.config["image_dir"], f"{first_photo_id}.jpg")

        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # Create a black image as a placeholder if not found
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # ViT processor returns a pixel_values tensor
        processed_image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        return {
            'text_input_ids': text_encoding['input_ids'].flatten(),
            'text_attention_mask': text_encoding['attention_mask'].flatten(),
            'image': processed_image,
            'label': torch.tensor(label, dtype=torch.long)
        }


# --- 5. Training and Evaluation Functions ---

def train_epoch(model, data_loader, optimizer, device, config):
    model.train()
    total_loss, total_cls_loss, total_con_loss, total_tri_loss = 0, 0, 0, 0
    total_correct = 0

    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['text_input_ids'].to(device)
        attention_mask = batch['text_attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        text_input = {'input_ids': input_ids, 'attention_mask': attention_mask}

        logits, l_con, l_triplet = model(text_input, images, labels)

        l_cls = F.cross_entropy(logits, labels)

        loss = l_cls + config["lambda1_con"] * l_con + config["lambda2_triplet"] * l_triplet

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += l_cls.item()
        total_con_loss += l_con.item()
        total_tri_loss += l_triplet.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    avg_con_loss = total_con_loss / len(data_loader)
    avg_tri_loss = total_tri_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)

    return accuracy, avg_loss, avg_cls_loss, avg_con_loss, avg_tri_loss


def eval_model(model, data_loader, device):
    model.eval()
    total_correct = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            text_input = {'input_ids': input_ids, 'attention_mask': attention_mask}

            logits, _, _ = model(text_input, images)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()

    accuracy = total_correct / len(data_loader.dataset)
    return accuracy


# --- 6. Main Execution Block ---

if __name__ == '__main__':
    # Load data
    df = pd.read_csv(CONFIG["data_path"])
    # Drop rows with no text
    df.dropna(subset=['content'], inplace=True)
    # Simple 80/20 split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    # Initialize tokenizer and image processor
    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    image_processor = ViTImageProcessor.from_pretrained(CONFIG["vit_model"])

    # Create datasets and dataloaders
    train_dataset = SpamReviewDataset(train_df, tokenizer, image_processor, CONFIG)
    val_dataset = SpamReviewDataset(val_df, tokenizer, image_processor, CONFIG, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # Initialize model and optimizer
    model = SR_CIBN(CONFIG).to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # Training loop
    for epoch in range(CONFIG["epochs"]):
        print(f"--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")

        train_acc, train_loss, cls_loss, con_loss, tri_loss = train_epoch(model, train_loader, optimizer,
                                                                          CONFIG["device"], CONFIG)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  -> CLS Loss: {cls_loss:.4f}, CON Loss: {con_loss:.4f}, TRI Loss: {tri_loss:.4f}")

        val_acc = eval_model(model, val_loader, CONFIG["device"])
        print(f"Val Acc: {val_acc:.4f}")

    print("Training complete.")
    # You can save the model here
    # torch.save(model.state_dict(), "sr_cibn_baseline.pth")