# train_sr_cibn.py
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# --- Import logging and tqdm ---
import logging
from tqdm import tqdm
import sys # For flushing stdout if needed
# --------------------------------

from sr_cibn_model import SR_CIBN, InfoNCELoss, compute_triplet_loss # Import your model and loss
from data_loader import MultimodalDataset # Import your data loader

# --- Configuration ---
DATA_CSV_PATH = '../../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv'
IMAGE_DIR = '../../spams_detection/spam_datasets/crawler/LA/images/'
MODEL_SAVE_PATH = 'sr_cibn_model.pth'
LEARNING_RATE = 0.001 # As per paper
NUM_EPOCHS = 10 # Start smaller for testing, paper is 120
FEATURE_DIM = 256 # As per paper
NUM_HEADS = 8 # As per paper
TEMPERATURE = 0.05 # For InfoNCE
LAMBDA1 = 0.3 # Weight for contrastive loss
LAMBDA2 = 0.5 # Weight for triplet loss
MARGIN_GAMMA = 0.5 # For triplet loss
THRESHOLD_TAU = 0.4 # For triplet loss clustering
SPARSITY_RATE_RHO = 0.6 # For patch selection

parser = argparse.ArgumentParser(description='Train SR-CIBN Model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
args = parser.parse_args()
BATCH_SIZE = args.batch_size

# --- Logging Configuration ---
logger = logging.getLogger('SR_CIBN_Trainer')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(c_format)

logger.addHandler(console_handler)
# -----------------------------

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# --- Data Loading ---
logger.info("Loading dataset...")
full_dataset = MultimodalDataset(DATA_CSV_PATH, IMAGE_DIR)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
logger.info(f"Dataset loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# --- Model, Loss, Optimizer ---
logger.info("Initializing model...")
model = SR_CIBN(feature_dim=FEATURE_DIM, num_heads=NUM_HEADS, num_classes=2)
model.to(device)

criterion_cls = nn.CrossEntropyLoss()
criterion_con = InfoNCELoss(temperature=TEMPERATURE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
logger.info("Model, loss functions, and optimizer initialized.")

# --- Metrics Calculation ---
def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    try:
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    except ValueError:
        logger.warning("Only one class present in batch predictions/labels, setting prec/rec/f1 to 0.")
        prec, rec, f1 = 0.0, 0.0, 0.0
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

    progress_desc = f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"
    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False, ncols=100, file=sys.stdout)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_tensor_spatial = batch['image_tensor_spatial'].to(device)
        image_tensor_freq = batch['image_tensor_freq'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, image_tensor_spatial, image_tensor_freq)
        logits = outputs['logits']
        features = outputs['features'] # [B, D]
        GI_part = outputs['GI_part'] # [B, D]
        GM_part = outputs['GM_part'] # [B, D]
        alpha_dynamic = outputs['alpha_dynamic'] # [B]
        ut_global = outputs['ut_global'] # [B, D]
        uv_global = outputs['uv_global'] # [B, D]
        uf_global = outputs['uf_global'] # [B, D]

        # --- Loss Calculation ---
        cls_loss = criterion_cls(logits, labels)

        # Contrastive Loss (Multi-view: t2v, v2t, v2f using global features)
        con_loss_t2v = criterion_con(ut_global, uv_global)
        con_loss_v2t = criterion_con(uv_global, ut_global) # Usually same as t2v, but let's compute both ways
        con_loss_v2f = criterion_con(uv_global, uf_global)
        con_loss_f2v = criterion_con(uf_global, uv_global) # Compute both ways
        # Average all contrastive losses
        con_loss = (con_loss_t2v + con_loss_v2t + con_loss_v2f + con_loss_f2v) / 4

        # Triplet Loss
        triplet_loss = compute_triplet_loss(features, labels, alpha_dynamic, threshold_tau, margin_gamma)

        # Total Loss
        total_batch_loss = cls_loss + lambda1 * con_loss + lambda2 * triplet_loss

        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_cls_loss += cls_loss.item()
        total_con_loss += con_loss.item()
        total_triplet_loss += triplet_loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update tqdm progress bar description with current average losses
        current_avg_loss = total_loss / (len(all_preds) / dataloader.batch_size)
        progress_bar.set_postfix({'Loss': f"{current_avg_loss:.4f}"})

    train_acc, train_prec, train_rec, train_f1 = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_con_loss = total_con_loss / num_batches
    avg_triplet_loss = total_triplet_loss / num_batches

    logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training Metrics - "
                f"Avg Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, "
                f"Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
    progress_bar.close()
    return avg_loss, train_acc, train_prec, train_rec, train_f1

# --- Validation Function ---
def validate_epoch(model, dataloader, criterion_cls, device, epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = len(dataloader)

    progress_desc = f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"
    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False, ncols=100, file=sys.stdout)

    with torch.no_grad():
        for batch in progress_bar:
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

            # Optional: Update progress bar postfix
            # current_val_loss = total_loss / (len(all_preds) / dataloader.batch_size)
            # progress_bar.set_postfix({'Val_Loss': f"{current_val_loss:.4f}"})

    val_loss = total_loss / num_batches
    val_acc, val_prec, val_rec, val_f1 = calculate_metrics(all_labels, all_preds)

    logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Metrics - "
                f"Avg Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, "
                f"Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
    progress_bar.close()
    return val_loss, val_acc, val_prec, val_rec, val_f1

# --- Main Training Loop ---
logger.info("Starting training loop...")
best_val_f1 = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
        model, train_loader, optimizer, criterion_cls, criterion_con, device,
        epoch, LAMBDA1, LAMBDA2, MARGIN_GAMMA, THRESHOLD_TAU
    )
    val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch(model, val_loader, criterion_cls, device, epoch)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logger.info(f"New best model saved with F1: {best_val_f1:.4f} (Epoch {epoch+1})")

logger.info("Training finished.")
logger.info(f"Best validation F1 score: {best_val_f1:.4f}")
