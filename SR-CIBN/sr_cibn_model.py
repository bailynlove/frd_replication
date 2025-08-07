# sr_cibn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, ViTModel, ViTConfig
import torchvision.models as models
from torchvision import transforms
import math

# --- Helper Modules ---

class CrossModalAttention(nn.Module):
    """Cross-Modal Attention Module (CMA)"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, kv):
        # q: [batch_size, seq_len_q, dim]
        # kv: [batch_size, seq_len_kv, dim] -> k, v
        B, Nq, D = q.shape
        _, Nkv, _ = kv.shape

        Q = self.q_proj(q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, Nq, dh]
        K = self.k_proj(kv).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, Nkv, dh]
        V = self.v_proj(kv).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, Nkv, dh]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale # [B, nh, Nq, Nkv]
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, nh, Nq, Nkv]

        out = torch.matmul(attn_weights, V) # [B, nh, Nq, dh]
        out = out.transpose(1, 2).contiguous().view(B, Nq, D) # [B, Nq, D]
        out = self.out_proj(out) # [B, Nq, D]
        return out

class GatedMultimodalUnit(nn.Module):
    """Gated Multimodal Unit (GMU)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.h1_proj = nn.Linear(dim, dim)
        self.h2_proj = nn.Linear(dim, dim)
        self.z_proj = nn.Linear(2 * dim, dim) # Takes concatenated input

    def forward(self, x1, x2):
        # x1, x2: [batch_size, dim]
        h1 = torch.tanh(self.h1_proj(x1)) # [B, D]
        h2 = torch.tanh(self.h2_proj(x2)) # [B, D]
        z = torch.sigmoid(self.z_proj(torch.cat([x1, x2], dim=-1))) # [B, D]

        out = z * h1 + (1 - z) * h2 # [B, D]
        return out

# --- Main SR-CIBN Model ---

class SR_CIBN(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, dropout=0.1, num_classes=2, vit_model_name='google/vit-base-patch16-224-in21k'):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        # --- 1. Unimodal Feature Extraction ---
        # Text Encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-chinese')
        self.text_dim = self.text_encoder.config.hidden_size
        self.text_projector = nn.Linear(self.text_dim, feature_dim)

        # Image Encoder (Spatial - ViT)
        config = ViTConfig.from_pretrained(vit_model_name)
        self.image_encoder_spatial = ViTModel.from_pretrained(vit_model_name, config=config)
        self.image_dim_spatial = self.image_encoder_spatial.config.hidden_size
        self.image_projector_spatial = nn.Linear(self.image_dim_spatial, feature_dim)

        # Image Encoder (Frequency - Placeholder, using InceptionV3 backbone)
        # Note: You need to implement DCT transformation in the data loading part
        # This assumes input is already DCT transformed or InceptionV3 handles it implicitly
        # For simplicity, using standard InceptionV3, assuming DCT preprocessing happens elsewhere
        self.image_encoder_freq = models.inception_v3(pretrained=True, transform_input=False)
        # Modify final layer to output features instead of classification
        self.image_encoder_freq.fc = nn.Identity() # Remove final classification layer
        self.image_dim_freq = 2048 # Standard output dim for InceptionV3 features
        self.image_projector_freq = nn.Linear(self.image_dim_freq, feature_dim)

        # Layer Normalization
        self.ln_text = nn.LayerNorm(feature_dim)
        self.ln_image_spatial = nn.LayerNorm(feature_dim)
        self.ln_image_freq = nn.LayerNorm(feature_dim)

        # --- 2. Hierarchical Multimodal Feature Fusion (MFF) ---
        self.cma1 = CrossModalAttention(feature_dim, num_heads)
        self.cma2 = CrossModalAttention(feature_dim, num_heads)
        self.cma3 = CrossModalAttention(feature_dim, num_heads)

        self.gmu1 = GatedMultimodalUnit(feature_dim)
        self.gmu2 = GatedMultimodalUnit(feature_dim)

        # --- 3. Global Consistency/Inconsistency Features Extraction ---
        self.patch_attention = CrossModalAttention(feature_dim, num_heads) # For patch interaction
        self.global_attention_consistent = CrossModalAttention(feature_dim, num_heads) # For final interaction
        self.global_attention_inconsistent = CrossModalAttention(feature_dim, num_heads) # For final interaction

        # MLP for local feature enhancement
        self.local_enhance_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # --- 4. Dynamic Fusion ---
        # Cosine similarity and fusion logic are handled in forward pass

        # --- 5. Classifier ---
        self.classifier = nn.Linear(feature_dim * 2, num_classes) # Concatenated GI and GM

        # --- 6. Loss Components (Defined outside for flexibility) ---
        # Contrastive Loss, Triplet Loss will be calculated separately

    def forward(self, text_input_ids, text_attention_mask, image_tensor_spatial, image_tensor_freq):
        # --- 1. Unimodal Feature Extraction ---
        # Text
        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        # Use [CLS] token representation
        ut_global = self.text_projector(text_outputs.last_hidden_state[:, 0, :]) # [B, D]
        ut = self.text_projector(text_outputs.last_hidden_state) # [B, Nt, D]
        ut = self.ln_text(ut)

        # Image Spatial (ViT)
        image_outputs_spatial = self.image_encoder_spatial(pixel_values=image_tensor_spatial)
        # Use [CLS] token for global representation, patch tokens for local
        uv_global = self.image_projector_spatial(image_outputs_spatial.last_hidden_state[:, 0, :]) # [B, D]
        uv = self.image_projector_spatial(image_outputs_spatial.last_hidden_state[:, 1:, :]) # [B, Nv, D] (exclude CLS)
        uv = self.ln_image_spatial(uv)

        # Image Frequency (InceptionV3 - assumes DCT preprocessing done)
        # Ensure image_tensor_freq is preprocessed correctly for InceptionV3
        image_outputs_freq = self.image_encoder_freq(image_tensor_freq)
        uf_global = self.image_projector_freq(image_outputs_freq) # [B, D]
        # For simplicity, treating frequency as a single global feature vector
        uf = uf_global.unsqueeze(1).expand(-1, uv.shape[1], -1) # [B, Nv, D] (broadcast to patch size)
        uf = self.ln_image_freq(uf)

        # --- 2. Hierarchical Multimodal Feature Fusion (MFF) ---
        # F1 = CMA(uv, ut)
        F1 = self.cma1(uv, ut) # [B, Nv, D] using uv as Q, ut as KV
        F1_global = F.adaptive_avg_pool1d(F1.transpose(1, 2), 1).squeeze(-1) # [B, D] Pool to get global-like

        # F2 = CMA(F1, uv)
        F2 = self.cma2(F1, uv) # [B, Nv, D]
        F2_global = F.adaptive_avg_pool1d(F2.transpose(1, 2), 1).squeeze(-1) # [B, D]

        # F3 = CMA(F2, uf)
        F3 = self.cma3(F2, uf) # [B, Nv, D]
        F3_global = F.adaptive_avg_pool1d(F3.transpose(1, 2), 1).squeeze(-1) # [B, D]

        # GMU(F3, F2) -> H
        H = self.gmu1(F3_global, F2_global) # [B, D]
        # GMU(H, F1) -> Otvf
        Otvf = self.gmu2(H, F1_global) # [B, D] Final Global Feature

        # --- 3. Global Consistency/Inconsistency Features Extraction ---
        # Compute attention scores for patches
        # Assume ut_global is text global, uv is image patches
        # Self-attention within image patches (approximated)
        # Cross-attention between image patches (uv) and text global (ut_global)
        # Expand ut_global for patch-wise comparison
        ut_global_expanded = ut_global.unsqueeze(1).expand(-1, uv.shape[1], -1) # [B, Nv, D]

        # Self-attention score (simplified using dot product)
        alpha_is = torch.einsum('bid,bjd->bij', uv, uv) / math.sqrt(self.feature_dim) # [B, Nv, Nv]
        alpha_is = F.softmax(alpha_is, dim=-1)
        # Use diagonal for self-score (or mean pooling across patches for global vg)
        vg_approx = F.adaptive_avg_pool1d(uv.transpose(1, 2), 1).squeeze(-1) # [B, D] Approx global image
        vg_expanded = vg_approx.unsqueeze(1).expand(-1, uv.shape[1], -1) # [B, Nv, D]

        # Cross-attention score (image patch to text global)
        alpha_icM = torch.einsum('bid,bd->bi', uv, ut_global) / math.sqrt(self.feature_dim) # [B, Nv]
        alpha_icM = F.softmax(alpha_icM, dim=-1) # [B, Nv] Consistent scores

        # Inconsistent score (using negative similarity)
        alpha_icI = torch.einsum('bid,bd->bi', uv, ut_global) / math.sqrt(self.feature_dim) # [B, Nv]
        alpha_icI = F.softmax(-alpha_icI, dim=-1) # [B, Nv] Inconsistent scores

        # Combine scores (paper uses self + cross, simplified here)
        alpha_iM = (alpha_icM + 0.5) / 2 # Placeholder, refine if needed
        alpha_iI = (alpha_icI + 0.5) / 2 # Placeholder, refine if needed

        # Sparse selection (placeholder, select top Ns patches)
        # For simplicity, let's assume we select top 50% patches based on scores
        Nv = uv.shape[1]
        Ns = max(1, Nv // 2) # Ensure at least 1 patch

        # Select patches for consistency
        _, top_M_indices = torch.topk(alpha_iM, Ns, dim=1) # [B, Ns]
        V_M_s = torch.gather(uv, 1, top_M_indices.unsqueeze(-1).expand(-1, -1, uv.shape[-1])) # [B, Ns, D]

        # Select patches for inconsistency
        _, top_I_indices = torch.topk(alpha_iI, Ns, dim=1) # [B, Ns]
        V_I_s = torch.gather(uv, 1, top_I_indices.unsqueeze(-1).expand(-1, -1, uv.shape[-1])) # [B, Ns, D]

        # Local Feature Enhancement (MLP + Residual)
        V_M_s_enhanced = self.local_enhance_mlp(V_M_s) + V_M_s # [B, Ns, D]
        V_I_s_enhanced = self.local_enhance_mlp(V_I_s) + V_I_s # [B, Ns, D]

        # Fusion of remaining patches (placeholder - simple mean)
        # In practice, use softmax weights from alpha_iM/I for remaining patches
        # Here, we'll just use the mean of selected enhanced patches for demonstration
        v_M_f = torch.mean(V_M_s_enhanced, dim=1) # [B, D]
        v_I_f = torch.mean(V_I_s_enhanced, dim=1) # [B, D]

        # Concatenate enhanced patches with fused features
        V_M_hat = torch.cat([V_M_s_enhanced, v_M_f.unsqueeze(1)], dim=1) # [B, Ns+1, D]
        V_I_hat = torch.cat([V_I_s_enhanced, v_I_f.unsqueeze(1)], dim=1) # [B, Ns+1, D]

        # Interact with global features Otvf
        # V_M_hat as Q, Otvf as KV (expand Otvf)
        Otvf_expanded = Otvf.unsqueeze(1).expand(-1, V_M_hat.shape[1], -1) # [B, Ns+1, D]
        Mall = self.global_attention_consistent(V_M_hat, Otvf_expanded) # [B, Ns+1, D]
        Mall = torch.mean(Mall, dim=1) # [B, D] Pool to single vector

        Iall = self.global_attention_inconsistent(V_I_hat, Otvf_expanded) # [B, Ns+1, D]
        Iall = torch.mean(Iall, dim=1) # [B, D] Pool to single vector

        # Obtain final Global Consistency (GM) and Inconsistency (GI) features
        GI = torch.cat([Iall, Otvf], dim=-1) # [B, 2*D]
        GM = torch.cat([Mall, Otvf], dim=-1) # [B, 2*D]

        # --- 4. Dynamic Fusion ---
        # Compute image-text matching degree (cosine similarity)
        # ut_global [B, D], uv_global [B, D]
        cos_sim = F.cosine_similarity(ut_global, uv_global, dim=1) # [B]
        sij = cos_sim

        # Transform sij to alpha
        pij = torch.sigmoid(sij) # [B]
        # Normalize pij to [0, 1] (assuming batch-wise normalization as per paper description)
        # Paper uses min/max over batch, but for stability, might use fixed range or instance norm
        # Simplified: Use sigmoid output directly as alpha (range [0,1])
        alpha = pij # [B]

        # Expand alpha for feature dimension
        alpha_expanded = alpha.unsqueeze(-1) # [B, 1]

        # Dynamic fusion F = alpha * GI + (1-alpha) * GM
        # Since GI and GM are [B, 2*D], we need to adjust or split Otvf
        # Let's redefine GM and GI to be [B, D] by interacting Mall/Iall with Otvf directly
        # Refine:
        GI_refined = Iall # [B, D] Inconsistency part
        GM_refined = Mall # [B, D] Consistency part
        alpha_expanded_d = alpha.unsqueeze(-1) # [B, 1]

        F_final = alpha_expanded_d * GI_refined + (1 - alpha_expanded_d) * GM_refined # [B, D]

        # --- 5. Classification ---
        logits = self.classifier(F_final) # [B, num_classes]

        # Return necessary features for loss calculation
        return {
            'logits': logits,
            'features': F_final, # Final fused feature [B, D]
            'GI': GI_refined,     # Global Inconsistency [B, D]
            'GM': GM_refined,     # Global Consistency [B, D]
            'alpha': alpha,       # Matching degree [B]
            'ut': ut_global,      # Text global [B, D]
            'uv': uv_global,      # Image spatial global [B, D]
            'uf': uf_global,      # Image frequency global [B, D]
            # Add other features needed for contrastive loss if computed separately
        }

# --- Loss Functions (Separate for clarity and flexibility) ---

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_a, features_b, labels=None):
        """
        Compute InfoNCE loss.
        features_a, features_b: [batch_size, feature_dim]
        labels: [batch_size] - Optional, for supervised contrastive (not used here for simplicity)
        """
        # Normalize features
        features_a = F.normalize(features_a, p=2, dim=1)
        features_b = F.normalize(features_b, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features_a, features_b.T) / self.temperature # [B, B]

        # Labels: positive pairs are on the diagonal
        batch_size = features_a.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=features_a.device)

        # Compute InfoNCE loss (t2v and v2t)
        loss_t2v = F.cross_entropy(sim_matrix, labels)
        loss_v2t = F.cross_entropy(sim_matrix.T, labels) # Transpose for v2t

        return (loss_t2v + loss_v2t) / 2

# Placeholder for Triplet Loss - needs clustering logic in training loop
# def compute_triplet_loss(features, labels, margin=0.5, alpha_values=None, threshold_tau=0.4):
#     # Implement triplet loss based on alpha_values and threshold_tau
#     # Split features into C1 (alpha >= tau) and C2 (alpha < tau)
#     # For each cluster, compute triplets and loss
#     # This is typically done in the training loop
#     pass
