# train_sr_cibn.py

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sr_cibn_model import SR_CIBN, InfoNCELoss # Import your model and loss
from data_loader import MultimodalDataset # Import your data loader

# --- Configuration ---
DATA_CSV_PATH = '../../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv'
IMAGE_DIR = '../../spams_detection/spam_datasets/crawler/LA/images/'
MODEL_SAVE_PATH = 'sr_cibn_model.pth'
LEARNING_RATE = 0.001 # As per paper
NUM_EPOCHS = 10 # paper is 120 epochs, but start smaller for testing
FEATURE_DIM = 256 # As per paper
NUM_HEADS = 8 # As per paper
TEMPERATURE = 0.05 # For InfoNCE
LAMBDA1 = 0.3 # Weight for contrastive loss
LAMBDA2 = 0.5 # Weight for triplet loss (placeholder)
MARGIN_GAMMA = 0.5 # For triplet loss
THRESHOLD_TAU = 0.4 # For triplet loss clustering
SPARSITY_RATE_RHO = 0.6 # For patch selection

parser = argparse.ArgumentParser(description='Train SR-CIBN Model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
args = parser.parse_args()
BATCH_SIZE = args.batch_size

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Loading ---
full_dataset = MultimodalDataset(DATA_CSV_PATH, IMAGE_DIR)
# Split dataset (adjust split sizes as needed)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Usually no shuffle for val

# --- Model, Loss, Optimizer ---
model = SR_CIBN(feature_dim=FEATURE_DIM, num_heads=NUM_HEADS, num_classes=2)
model.to(device)

# Loss functions
criterion_cls = nn.CrossEntropyLoss()
criterion_con = InfoNCELoss(temperature=TEMPERATURE)
# Triplet loss function needs to be defined or integrated into training loop logic
# We'll calculate it inline for now

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Consider learning rate scheduler if needed

# --- Metrics Calculation ---
def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return acc, prec, rec, f1

# --- Training Function (Single Epoch) ---
def train_epoch(model, dataloader, optimizer, criterion_cls, criterion_con, device, epoch, lambda1, lambda2, margin_gamma, threshold_tau):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_con_loss = 0
    total_triplet_loss = 0
    all_preds = []
    all_labels = []
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_tensor_spatial = batch['image_tensor_spatial'].to(device)
        image_tensor_freq = batch['image_tensor_freq'].to(device) # Ensure correct preprocessing
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask, image_tensor_spatial, image_tensor_freq)
        logits = outputs['logits']
        features = outputs['features']
        GI = outputs['GI']
        GM = outputs['GM']
        alpha = outputs['alpha']
        ut = outputs['ut']
        uv = outputs['uv']
        uf = outputs['uf'] # Add uf if needed for v2f contrastive

        # --- Loss Calculation ---
        # 1. Classification Loss
        cls_loss = criterion_cls(logits, labels)

        # 2. Contrastive Loss (Multi-view: t2v, v2t, v2f)
        # Example: t2v and v2t using ut and uv
        con_loss_t2v_v2t = criterion_con(ut, uv) # Simplified, might need averaging positive/negative pairs properly
        # Example: v2f using uv and uf (if uf is patch-wise or global processed correctly)
        # con_loss_v2f = criterion_con(uv_global_processed, uf_global) # Define uv_global_processed, uf_global appropriately
        con_loss_v2f = torch.tensor(0.0, device=device) # Placeholder
        con_loss = (con_loss_t2v_v2t + con_loss_v2f) / 2 # Average or sum as per paper

        # 3. Triplet Loss (Joint Learning - requires clustering)
        # Cluster based on alpha and threshold_tau
        # This is a simplified version, assumes features are [B, D]
        triplet_loss = torch.tensor(0.0, device=device) # Initialize
        # --- Triplet Loss Logic ---
        # Separate features into C1 (alpha >= tau) and C2 (alpha < tau)
        mask_c1 = alpha >= threshold_tau
        mask_c2 = alpha < threshold_tau

        # Function to compute triplet loss for a cluster
        def compute_cluster_triplet_loss(cluster_features, cluster_labels, margin):
            loss = 0.0
            if len(cluster_features) < 2: # Need at least 2 samples
                return torch.tensor(0.0, device=device)
            # For each anchor in the cluster
            for i in range(len(cluster_features)):
                anchor_feat = cluster_features[i]
                anchor_label = cluster_labels[i]

                # Find positive and negative samples
                pos_mask = (cluster_labels == anchor_label) & (torch.arange(len(cluster_labels), device=device) != i)
                neg_mask = (cluster_labels != anchor_label)

                if not pos_mask.any() or not neg_mask.any():
                    continue # Skip if no pos or neg

                pos_features = cluster_features[pos_mask]
                neg_features = cluster_features[neg_mask]

                # Use the closest positive and hardest negative (simplified)
                # Positive distance
                pos_distances = torch.norm(anchor_feat.unsqueeze(0) - pos_features, dim=1) # [num_pos]
                pos_dist = torch.min(pos_distances) # Closest positive

                # Negative distance
                neg_distances = torch.norm(anchor_feat.unsqueeze(0) - neg_features, dim=1) # [num_neg]
                neg_dist = torch.min(neg_distances) # Hardest negative (closest negative)

                # Triplet loss
                loss += torch.relu(pos_dist - neg_dist + margin)
            return loss / len(cluster_features) if len(cluster_features) > 0 else torch.tensor(0.0, device=device)

        # Apply to C1 and C2
        if mask_c1.any():
            features_c1 = features[mask_c1]
            labels_c1 = labels[mask_c1]
            triplet_loss_c1 = compute_cluster_triplet_loss(features_c1, labels_c1, margin_gamma)
            triplet_loss += triplet_loss_c1

        if mask_c2.any():
            features_c2 = features[mask_c2]
            labels_c2 = labels[mask_c2]
            triplet_loss_c2 = compute_cluster_triplet_loss(features_c2, labels_c2, margin_gamma)
            triplet_loss += triplet_loss_c2

        # Total Loss
        total_batch_loss = cls_loss + lambda1 * con_loss + lambda2 * triplet_loss

        # Backward pass
        total_batch_loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += total_batch_loss.item()
        total_cls_loss += cls_loss.item()
        total_con_loss += con_loss.item()
        total_triplet_loss += triplet_loss.item() # Might be tensor(0)

        # Predictions
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Print progress (optional)
        if batch_idx % 10 == 0:
             print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{num_batches}], "
                   f"Loss: {total_batch_loss.item():.4f}, CLS: {cls_loss.item():.4f}, "
                   f"CON: {con_loss.item():.4f}, TPL: {triplet_loss.item():.4f}")

    # Calculate metrics for the epoch
    train_acc, train_prec, train_rec, train_f1 = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_con_loss = total_con_loss / num_batches
    avg_triplet_loss = total_triplet_loss / num_batches # Might be 0 if triplet not computed

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training Metrics - "
          f"Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, "
          f"Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
    return avg_loss, train_acc, train_prec, train_rec, train_f1

# --- Validation Function ---
def validate_epoch(model, dataloader, criterion_cls, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_tensor_spatial = batch['image_tensor_spatial'].to(device)
            image_tensor_freq = batch['image_tensor_freq'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, image_tensor_spatial, image_tensor_freq)
            logits = outputs['logits']
            loss = criterion_cls(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / num_batches
    val_acc, val_prec, val_rec, val_f1 = calculate_metrics(all_labels, all_preds)
    print(f"Validation Metrics - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
          f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
    return val_loss, val_acc, val_prec, val_rec, val_f1

# --- Main Training Loop ---
best_val_f1 = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
        model, train_loader, optimizer, criterion_cls, criterion_con, device,
        epoch, LAMBDA1, LAMBDA2, MARGIN_GAMMA, THRESHOLD_TAU
    )
    val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch(model, val_loader, criterion_cls, device)

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"New best model saved with F1: {best_val_f1:.4f}")

print("Training finished.")

# --- Final Evaluation on Test Set (Optional) ---
# You would load the best model and run it on a held-out test set
# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# test_loss, test_acc, test_prec, test_rec, test_f1 = validate_epoch(model, test_loader, criterion_cls, device)
# print(f"Final Test Metrics - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
#       f"Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")
