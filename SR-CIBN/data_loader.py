# data_loader.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import cv2 # For DCT

# --- Image Transformations ---
# Standard ViT transform
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # ViT normalization
])

# Standard InceptionV3 transform (for frequency domain - adjust if DCT is applied separately)
inception_transform = transforms.Compose([
    transforms.Resize(299), # InceptionV3 expects 299x299
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# Function to apply DCT (example, might be integrated into transform or done offline)
def apply_dct_transform(image_tensor):
    # Convert tensor to numpy (C, H, W)
    image_np = image_tensor.permute(1, 2, 0).numpy()
    # Convert to grayscale if needed (DCT often applied to grayscale)
    if image_np.shape[2] == 3:
        image_gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image_gray = (image_np.squeeze() * 255).astype(np.uint8)

    # Apply DCT
    dct_coeffs = cv2.dct(np.float32(image_gray))
    # Normalize DCT coefficients (optional, but common)
    dct_coeffs_normalized = cv2.normalize(dct_coeffs, None, 0, 255, cv2.NORM_MINMAX)
    # Convert back to tensor (H, W) -> (1, H, W) for single channel input
    dct_tensor = torch.tensor(dct_coeffs_normalized, dtype=torch.float32).unsqueeze(0) / 255.0
    return dct_tensor

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer_name='bert-base-uncased', max_length=128):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.vit_transform = vit_transform
        self.inception_transform = inception_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['content']) # Assuming 'content' is your text column
        try:
            label = int(row['is_recommended'])
        except KeyError:
            # Example: Define label based on rating or is_recommended
            # Adjust this logic based on your data definition of fake/real
            label = 1 if row.get('is_recommended', 0) == 1 else 0 # Placeholder

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze() # Remove batch dim
        attention_mask = encoding['attention_mask'].squeeze()

        # Load image (use first photo_id if multiple)
        photo_ids_str = str(row.get('photo_ids', '')) if not pd.isna(row.get('photo_ids')) else ''
        photo_id = photo_ids_str.split('#')[0] if photo_ids_str else None
        image_tensor_spatial = torch.zeros(3, 224, 224) # Default if no image
        image_tensor_freq = torch.zeros(3, 299, 299) # Default if no image (DCT often single channel)

        if photo_id:
            img_path = os.path.join(self.image_dir, f"{photo_id}.jpg")
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor_spatial = self.vit_transform(image)
                # For frequency domain, apply transform and then DCT
                image_tensor_freq_raw = self.inception_transform(image) # [3, 299, 299]
                # Apply DCT (assuming grayscale DCT, take one channel or convert)
                # This is a simplified approach, might need refinement
                # Apply DCT to one channel or grayscale version
                image_tensor_freq_dct_single = apply_dct_transform(image_tensor_freq_raw) # [1, 299, 299]
                image_tensor_freq = image_tensor_freq_dct_single.repeat(3, 1, 1)  # [3, H, W]
                # If you want to use the full 3-channel DCT processed image, adjust accordingly
                # Or, if InceptionV3 is supposed to process the DCT image directly, just use inception_transform
                # For now, let's assume apply_dct_transform gives the correct format for InceptionV3 freq encoder
                 # image_tensor_freq = inception_transform(Image.fromarray(...dct_processed...))
                 # Placeholder: Just use the raw transformed image for freq encoder for now
                 # image_tensor_freq = image_tensor_freq_raw # If InceptionV3 handles it

            except FileNotFoundError:
                print(f"Warning: Image not found at {row['review_id']}:{photo_ids_str}. Using zero tensor.")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Using zero tensor.")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_tensor_spatial': image_tensor_spatial,
            'image_tensor_freq': image_tensor_freq, # Ensure this is correctly preprocessed for freq encoder
            'label': torch.tensor(label, dtype=torch.long)
        }

# Example usage (not run here)
# dataset = MultimodalDataset('../spams_dataset/LA/outputs/full_data_0731_aug_4.csv',
#                             '../spams_dataset/LA/images/')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
