# -*- coding: utf-8 -*-
"""
Replication of "Unmasking deception: a CNN and adaptive PSO approach"

This script implements the core methodology of the paper:
1.  **Feature Extraction**: Uses pre-trained GloVe embeddings to generate
    sentence vectors for each review.
2.  **Feature Selection (APSO)**: A simplified Particle Swarm Optimization (PSO)
    is implemented to select the most informative feature subset.
    - The fitness of each particle (feature subset) is evaluated by training a
      fast Logistic Regression model and measuring its F1-score.
3.  **Final Classification (CNN)**: A Convolutional Neural Network (CNN) is
    trained using only the best features selected by PSO.
4.  **Data Balancing**: Continues to use RandomOversampling on the training set.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import os
import requests
import zipfile
import warnings

warnings.filterwarnings('ignore')

# --- 1. 全局配置 ---
class Config:
    CSV_PATH = '../spams_dataset/LA/outputs/full_data_0731_aug_4.csv'
    TEXT_COLUMN = 'content'
    LABEL_COLUMN = 'is_recommended'
    POSITIVE_CLASS_LABEL = 1

    # GloVe
    GLOVE_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'
    GLOVE_ZIP_PATH = './glove.6B.zip'
    GLOVE_DIR = './glove.6B/'
    EMBEDDING_DIM = 100 # 我们使用100维的GloVe

    # APSO
    N_PARTICLES = 20    # 粒子数量
    N_ITERATIONS = 30   # 迭代次数
    INERTIA = 0.5       # 惯性权重
    C1, C2 = 1.5, 1.5   # 加速常数

    # CNN
    BATCH_SIZE, EPOCHS, LEARNING_RATE = 64, 10, 1e-3
    
    RANDOM_STATE = 42
    MODEL_SAVE_PATH = "./best_cnn_apso_model.pth"

# --- 2. 设备检查 ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Using device: {device} ---")


# --- 3. GloVe词向量加载与文本处理 ---
def download_and_unzip_glove(url, zip_path, extract_path):
    if not os.path.exists(extract_path):
        print("GloVe embeddings not found. Downloading...")
        if not os.path.exists(zip_path):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        print("Unzipping GloVe...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("GloVe ready.")
    else:
        print("GloVe embeddings found.")

def load_glove_embeddings(path, dim):
    embeddings_index = {}
    with open(os.path.join(path, f'glove.6B.{dim}d.txt'), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def text_to_vector(text, embeddings, dim):
    words = str(text).lower().split()
    word_vectors = [embeddings.get(word, np.zeros(dim)) for word in words]
    if not word_vectors:
        return np.zeros(dim)
    return np.mean(word_vectors, axis=0)

# --- 4. PSO 特征选择 ---
def pso_feature_selection(X_train, y_train, X_val, y_val, config):
    n_features = X_train.shape[1]
    
    # 初始化粒子位置 (0/1向量) 和速度
    positions = np.random.randint(2, size=(config.N_PARTICLES, n_features))
    velocities = np.random.rand(config.N_PARTICLES, n_features)
    
    personal_best_positions = positions.copy()
    personal_best_scores = np.zeros(config.N_PARTICLES)
    
    global_best_position = np.zeros(n_features)
    global_best_score = -1.0

    print("\n--- Starting PSO Feature Selection ---")
    for iteration in tqdm(range(config.N_ITERATIONS), desc="PSO Iterations"):
        for i in range(config.N_PARTICLES):
            selected_features_mask = positions[i, :] > 0.5
            
            # 如果没有选中任何特征，给一个极差的分数
            if not np.any(selected_features_mask):
                fitness = 0.0
            else:
                X_train_subset = X_train[:, selected_features_mask]
                X_val_subset = X_val[:, selected_features_mask]
                
                # 使用快速的分类器评估特征子集的质量
                model = LogisticRegression(random_state=config.RANDOM_STATE, max_iter=200)
                model.fit(X_train_subset, y_train)
                y_val_pred = model.predict(X_val_subset)
                fitness = f1_score(y_val, y_val_pred, average='binary', zero_division=0)

            # 更新个体最优
            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = positions[i].copy()

        # 更新全局最优
        best_particle_idx = np.argmax(personal_best_scores)
        if personal_best_scores[best_particle_idx] > global_best_score:
            global_best_score = personal_best_scores[best_particle_idx]
            global_best_position = personal_best_positions[best_particle_idx].copy()

        # 更新粒子速度和位置
        r1, r2 = np.random.rand(), np.random.rand()
        velocities = (config.INERTIA * velocities +
                      config.C1 * r1 * (personal_best_positions - positions) +
                      config.C2 * r2 * (global_best_position - positions))
        
        # 将速度转换为位置更新的概率 (Sigmoid)
        sigmoid = 1 / (1 + np.exp(-velocities))
        positions = (np.random.rand(config.N_PARTICLES, n_features) < sigmoid).astype(int)

    print(f"PSO finished. Best F1-score found: {global_best_score:.4f}")
    return global_best_position > 0.5


# --- 5. CNN 模型与数据加载 ---
class CnnDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        # 计算卷积和池化后的维度
        conv_output_size = (input_dim - 7 + 6) + 1 # Conv1d output size
        pool_output_size = conv_output_size // 2
        self.fc1 = nn.Linear(64 * pool_output_size, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = x.unsqueeze(1) # [B, D] -> [B, 1, D]
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 6. 主执行流程 ---
if __name__ == '__main__':
    # Step 1: 加载和准备数据
    download_and_unzip_glove(Config.GLOVE_URL, Config.GLOVE_ZIP_PATH, Config.GLOVE_DIR)
    glove_embeddings = load_glove_embeddings(Config.GLOVE_DIR, Config.EMBEDDING_DIM)
    
    df = pd.read_csv(Config.CSV_PATH)
    df['vector'] = df[Config.TEXT_COLUMN].apply(lambda x: text_to_vector(x, glove_embeddings, Config.EMBEDDING_DIM))
    
    X = np.array(df['vector'].tolist())
    y = df[Config.LABEL_COLUMN].values

    # Step 2: 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=Config.RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=Config.RANDOM_STATE, stratify=y_test)

    # Step 3: 对训练集进行过采样
    print("\nApplying RandomOversampling...")
    ros = RandomOverSampler(random_state=Config.RANDOM_STATE)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # Step 4: 运行PSO进行特征选择
    best_features_mask = pso_feature_selection(X_train_res, y_train_res, X_val, y_val, Config)
    n_selected = np.sum(best_features_mask)
    print(f"Selected {n_selected} features out of {X.shape[1]}.")
    
    # 使用最优特征子集
    X_train_final = X_train_res[:, best_features_mask]
    X_val_final = X_val[:, best_features_mask]
    X_test_final = X_test[:, best_features_mask]
    
    # Step 5: 训练最终的CNN分类器
    print("\n--- Training final CNN model with selected features ---")
    train_dataset = CnnDataset(X_train_final, y_train_res)
    val_dataset = CnnDataset(X_val_final, y_val)
    test_dataset = CnnDataset(X_test_final, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    model = SimpleCNN(input_dim=n_selected).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val_f1 = -1.0
    for epoch in range(Config.EPOCHS):
        model.train()
        for features, labels in tqdm(train_loader, desc=f"CNN Epoch {epoch+1}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds_val, all_labels_val = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.numpy())
        
        val_f1 = f1_score(all_labels_val, all_preds_val, average='binary', zero_division=0)
        print(f"Validation F1-score: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✨ New best CNN model saved.")
            
    # Step 6: 最终评估
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()
    
    all_preds_test, all_labels_test = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.numpy())
            
    accuracy = accuracy_score(all_labels_test, all_preds_test)
    precision = precision_score(all_labels_test, all_preds_test, average='binary', zero_division=0)
    recall = recall_score(all_labels_test, all_preds_test, average='binary', zero_division=0)
    f1 = f1_score(all_labels_test, all_preds_test, average='binary', zero_division=0)
    
    print("\n--- FINAL CNN+APSO PERFORMANCE ---")
    print("Metrics are calculated for the positive class (label=1)")
    print("="*40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("="*40)