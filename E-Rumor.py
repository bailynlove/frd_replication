import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split
import argparse
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# --- NEW: Setup argparse ---
parser = argparse.ArgumentParser(description='Train E-Rumor Model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction.')
parser.add_argument('--k_shot', type=int, default=5, help='Number of support samples per class (K in K-shot).')
parser.add_argument('--n_query', type=int, default=15, help='Number of query samples per class.')
parser.add_argument('--n_episodes', type=int, default=100, help='Number of few-shot evaluation episodes.')
parser.add_argument('--k_nearest', type=int, default=3,
                    help='Number of nearest base distributions to use for calibration.')
parser.add_argument('--n_generate', type=int, default=100,
                    help='Number of samples to generate per calibrated distribution.')
args = parser.parse_args()

# --- 1. Configuration ---
CONFIG = {
    "data_path": "../spams_detection/spam_datasets/crawler/LA/outputs/full_data_0731_aug_4.csv",
    "image_dir": "../spams_detection/spam_datasets/crawler/LA/images/",  # Not used, but kept for consistency
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_model": "bert-base-uncased",
    "feature_dim": 768,
    "mlp_hidden_dim": 256,
    "num_classes": 2,
    "base_event_ratio": 0.7,  # 70% of businesses will be base events
}

print(f"Using device: {CONFIG['device']}")
print(f"Running {args.n_episodes} episodes of {args.k_shot}-shot learning.")


# --- 2. Feature Extractor and Dataset ---
class FeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token embedding as the feature
        return outputs.last_hidden_state[:, 0, :]


class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['content'])
        label = int(row['is_recommended'])
        encoding = self.tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# --- 3. The E-Rumor Framework ---
class ERumor:
    def __init__(self, feature_extractor, config, args):
        self.feature_extractor = feature_extractor.to(config["device"])
        self.feature_extractor.eval()  # Feature extractor is pre-trained and frozen
        self.config = config
        self.args = args
        self.device = config["device"]
        self.base_distributions = defaultdict(dict)  # Structure: {event_id: {class_id: (mean, cov)}}
        self.classifier = None

    def calculate_base_distributions(self, base_df, tokenizer):
        print("Calculating distributions for base events...")
        base_events = base_df['biz_alias'].unique()
        for event_id in tqdm(base_events, desc="Processing Base Events"):
            event_df = base_df[base_df['biz_alias'] == event_id]
            event_dataset = TextDataset(event_df, tokenizer)
            event_loader = DataLoader(event_dataset, batch_size=self.args.batch_size)

            features_by_class = defaultdict(list)
            with torch.no_grad():
                for batch in event_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label']
                    features = self.feature_extractor(input_ids, attention_mask)
                    for i in range(features.shape[0]):
                        features_by_class[labels[i].item()].append(features[i].cpu())

            for class_id, feats in features_by_class.items():
                if len(feats) > 1:
                    feats_tensor = torch.stack(feats)
                    mean = feats_tensor.mean(dim=0)
                    # Add small epsilon for numerical stability
                    cov = torch.cov(feats_tensor.T) + torch.eye(self.config["feature_dim"]) * 1e-6
                    self.base_distributions[event_id][class_id] = (mean, cov)

    def calibrate_and_generate(self, support_features, support_labels):
        generated_features = []
        generated_labels = []

        for x_nj, y_j in zip(support_features, support_labels):
            y_j = y_j.item()

            # Find k-nearest base distributions for class y_j
            similarities = []
            for event_id, class_dists in self.base_distributions.items():
                if y_j in class_dists:
                    mean_ij, _ = class_dists[y_j]
                    sim = F.cosine_similarity(x_nj.cpu().unsqueeze(0), mean_ij.unsqueeze(0)).item()
                    similarities.append(((event_id, y_j), sim))

            similarities.sort(key=lambda item: item[1], reverse=True)
            k_nearest = similarities[:self.args.k_nearest]

            if not k_nearest: continue

            # Calibrate mean and covariance (Eq. 7 & 8)
            nearest_means = torch.stack([self.base_distributions[eid][cid][0] for (eid, cid), _ in k_nearest])
            nearest_covs = torch.stack([self.base_distributions[eid][cid][1] for (eid, cid), _ in k_nearest])

            calibrated_mean = (nearest_means.sum(dim=0) + x_nj.cpu()) / (len(k_nearest) + 1)
            calibrated_cov = nearest_covs.mean(dim=0)

            # Generate new samples (Eq. 10)
            try:
                distribution = MultivariateNormal(calibrated_mean, covariance_matrix=calibrated_cov)
                generated_samples = distribution.sample((self.args.n_generate,))
                generated_features.append(generated_samples)
                generated_labels.extend([y_j] * self.args.n_generate)
            except Exception as e:
                # If covariance is not positive definite, skip generation for this sample
                # print(f"Warning: Could not generate samples for class {y_j}. {e}")
                pass

        if not generated_features:
            return None, None

        return torch.cat(generated_features), torch.tensor(generated_labels, dtype=torch.long)

    def fit(self, support_features, support_labels):
        # Create the simple MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.config["feature_dim"], self.config["mlp_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(self.config["mlp_hidden_dim"], self.config["num_classes"])
        ).to(self.device)

        gen_feats, gen_labels = self.calibrate_and_generate(support_features, support_labels)

        if gen_feats is None:  # Handle case where no samples could be generated
            train_feats = support_features
            train_labels = support_labels
        else:
            train_feats = torch.cat([support_features, gen_feats.to(self.device)])
            train_labels = torch.cat([support_labels, gen_labels.to(self.device)])

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)

        # Train the MLP for a few epochs
        self.classifier.train()
        for _ in range(20):  # Small number of epochs is usually sufficient
            optimizer.zero_grad()
            logits = self.classifier(train_feats)
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            optimizer.step()

    def predict(self, query_features):
        if self.classifier is None:
            raise RuntimeError("Classifier has not been fitted. Call .fit() first.")
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(query_features)
            preds = torch.argmax(logits, dim=1)
        return preds


# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Load data and feature extractor
    df = pd.read_csv(CONFIG["data_path"])
    df.dropna(subset=['content', 'biz_alias'], inplace=True)
    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    feature_extractor = FeatureExtractor(CONFIG["bert_model"])

    # 2. Split events into base and novel
    all_events = df['biz_alias'].unique()
    base_event_ids, novel_event_ids = train_test_split(all_events, test_size=1.0 - CONFIG["base_event_ratio"],
                                                       random_state=42)
    base_df = df[df['biz_alias'].isin(base_event_ids)]
    novel_df = df[df['biz_alias'].isin(novel_event_ids)]

    print(f"Total businesses (events): {len(all_events)}")
    print(f"Base events: {len(base_event_ids)}, Novel events: {len(novel_event_ids)}")

    # 3. Instantiate and "train" E-Rumor by calculating base distributions
    e_rumor = ERumor(feature_extractor, CONFIG, args)
    e_rumor.calculate_base_distributions(base_df, tokenizer)

    # 4. Run few-shot evaluation episodes
    episode_accuracies = []
    print(f"\nStarting {args.n_episodes} few-shot evaluation episodes...")
    for episode in tqdm(range(args.n_episodes)):
        # a. Select a novel event
        novel_event_id = np.random.choice(novel_event_ids)
        event_df = novel_df[novel_df['biz_alias'] == novel_event_id]

        # b. Sample support and query sets
        support_df_list, query_df_list = [], []
        available_classes = event_df['is_recommended'].unique()

        # Ensure we have at least k_shot + n_query samples for each class
        valid_classes = [c for c in available_classes if
                         event_df['is_recommended'].value_counts().get(c, 0) >= args.k_shot + args.n_query]

        if len(valid_classes) < CONFIG["num_classes"]:
            continue  # Skip episode if not enough data

        for class_id in valid_classes:
            class_df = event_df[event_df['is_recommended'] == class_id]
            # Sample without replacement
            samples = class_df.sample(n=args.k_shot + args.n_query, replace=False)
            support_df_list.append(samples.head(args.k_shot))
            query_df_list.append(samples.tail(args.n_query))

        if not support_df_list: continue

        support_df = pd.concat(support_df_list)
        query_df = pd.concat(query_df_list)

        # c. Extract features
        support_dataset = TextDataset(support_df, tokenizer)
        query_dataset = TextDataset(query_df, tokenizer)

        with torch.no_grad():
            support_feats = feature_extractor(support_dataset[:]['input_ids'].to(CONFIG["device"]),
                                              support_dataset[:]['attention_mask'].to(CONFIG["device"]))
            support_labels = support_dataset[:]['label'].to(CONFIG["device"])
            query_feats = feature_extractor(query_dataset[:]['input_ids'].to(CONFIG["device"]),
                                            query_dataset[:]['attention_mask'].to(CONFIG["device"]))
            query_labels = query_dataset[:]['label'].to(CONFIG["device"])

        # d. Fit the classifier on the support set
        e_rumor.fit(support_feats, support_labels)

        # e. Predict on the query set and calculate accuracy
        predictions = e_rumor.predict(query_feats)
        accuracy = (predictions == query_labels).float().mean().item()
        episode_accuracies.append(accuracy)

    # 5. Report final results
    if episode_accuracies:
        mean_accuracy = np.mean(episode_accuracies) * 100
        std_accuracy = np.std(episode_accuracies) * 100
        print("\n--- Few-Shot Evaluation Results ---")
        print(f"Average Accuracy over {len(episode_accuracies)} episodes: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
        print("------------------------------------")
    else:
        print(
            "\nCould not run any evaluation episodes. Check if novel events have enough data for K-shot + N-query setting.")