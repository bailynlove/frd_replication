import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import torch
import torch.nn.functional as F
from collections import Counter # <--- 在这里导入Counter

# NLP & ML Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.stats import entropy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from imblearn.over_sampling import SMOTE

# --- 0. 环境准备与设备检测 ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# --- 1. 数据加载与预处理 (无变化) ---
def load_and_prepare_data(filepath='full_data_0617.csv'):
    print(f"Step 1: Loading and preparing data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found. Please place it in the same directory as the script.")
    df = pd.read_csv(filepath)
    expected_cols = ['review_id', 'content', 'date', 'rating', 'polarity', 'subjectivity', 'is_recommended']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Error: The expected column '{col}' is missing from the CSV file.")
    df.dropna(subset=['content', 'date', 'rating'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['label'] = 1 - df['is_recommended']
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Fake review (is_recommended=0) proportion: {df['label'].mean():.2%}")
    print("NOTE: 'review_id' and 'subjectivity' columns are present but not used in this paper's feature set.")
    return df

# --- 2. 文本预处理与向量化 (无变化) ---
def preprocess_text(text):
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def train_and_vectorize_doc2vec(df):
    print("Step 2.1: Preprocessing text and training Doc2Vec model (CPU-based)...")
    df['tokens'] = df['content'].apply(preprocess_text)
    tagged_data = [TaggedDocument(words=row['tokens'], tags=[i]) for i, row in df.iterrows()]
    model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=-1, epochs=20, dm=1)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    print("Step 2.2: Vectorizing reviews using the trained model...")
    vectors = np.array([model.infer_vector(row['tokens']) for i, row in df.iterrows()])
    return vectors, model, df

# --- 3. 特征提取 (Stage 2) ---
def calculate_suspicion_degree(df):
    print("Step 3.1: Calculating the core 'Suspicion Degree' feature (CPU-based)...")
    daily_stats = df.groupby(df['date'].dt.date).agg(num_reviews=('content', 'size'), mean_score=('rating', 'mean')).reset_index()
    entropies = []
    for date, group in df.groupby(df['date'].dt.date):
        counts = group['rating'].value_counts(normalize=True)
        dist = [counts.get(i, 0) for i in range(1, 6)]
        entropies.append({'date': date, 'entropy': entropy(dist, base=2)})
    entropy_df = pd.DataFrame(entropies)
    time_series_df = pd.merge(daily_stats, entropy_df, on='date')
    X_time_series = time_series_df[['mean_score', 'entropy', 'num_reviews']].values
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', n_jobs=-1)
    time_series_df['is_outlier'] = lof.fit_predict(X_time_series)
    suspicious_dates = time_series_df[time_series_df['is_outlier'] == -1]['date'].tolist()
    df['suspicion_degree'] = df['date'].dt.date.isin(suspicious_dates).astype(int)
    print(f"Identified {len(suspicious_dates)} suspicious days with high review burst activity.")
    return df

def extract_other_features(df, doc2vec_vectors):
    """提取文本和行为特征, 使用MPS加速F15的计算"""
    print("Step 3.2: Extracting other linguistic and behavioral features...")
    features = pd.DataFrame(index=df.index)

    features['F1_emotional_intensity'] = df['polarity']
    
    # *** 修复点在这里 ***
    def pos_proportions(tokens):
        if not tokens: return 0, 0, 0, 0
        # 使用 collections.Counter, 而不是 nltk.Counter
        counts = Counter(tag for word, tag in nltk.pos_tag(tokens)) 
        total = len(tokens)
        if total == 0: return 0, 0, 0, 0
        nouns = sum(counts[tag] for tag in counts if tag.startswith('NN')) / total
        adjectives = sum(counts[tag] for tag in counts if tag.startswith('JJ')) / total
        verbs = sum(counts[tag] for tag in counts if tag.startswith('VB')) / total
        pronouns = sum(counts[tag] for tag in counts if tag.startswith('PRP')) / total
        return nouns, adjectives, verbs, pronouns

    pos_feats = df['tokens'].apply(lambda x: pd.Series(pos_proportions(x), index=['F5_nouns', 'F6_adjectives', 'F10_verbs', 'F12_pronouns']))
    features = pd.concat([features, pos_feats], axis=1)
    features['F14_text_length'] = df['content'].str.len()
    features['F16_score'] = df['rating']

    if device.type == 'mps':
        print("Calculating F15: Text Similarity using MPS (GPU)...")
        vecs_tensor = torch.from_numpy(doc2vec_vectors).to(device)
        vecs_tensor = F.normalize(vecs_tensor, p=2, dim=1)
        sim_matrix_tensor = torch.matmul(vecs_tensor, vecs_tensor.T)
        sim_matrix = sim_matrix_tensor.cpu().numpy()
    else:
        print("Calculating F15: Text Similarity using scikit-learn (CPU)...")
        sim_matrix = sklearn_cosine_similarity(doc2vec_vectors)

    np.fill_diagonal(sim_matrix, 0)
    features['F15_text_similarity'] = sim_matrix.mean(axis=1)

    print("NOTE: Features F17-F20 are excluded as they are not available.")
    features['F24_suspicion_degree'] = df['suspicion_degree']
    features = features.fillna(0)
    return features


# --- 4. 分类与评估 (无变化) ---
def train_and_evaluate(X, y):
    print("\nStep 4: Training and Evaluating the Model (CPU-based)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = min(13, X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA completed. Using {n_components} components, explaining {np.sum(pca.explained_variance_ratio_):.2%} of variance.")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Class distribution before SMOTE in training set: Genuine(0)={np.sum(y_train==0)}, Fake(1)={np.sum(y_train==1)}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE in training set:  Genuine(0)={np.sum(y_train_res==0)}, Fake(1)={np.sum(y_train_res==1)}")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    print("\n--- Evaluation Report ---")
    print(f"Target: Predicting 'Fake' reviews (label=1, which corresponds to is_recommended=0)")
    print(classification_report(y_test, y_pred, target_names=['Genuine (0)', 'Fake (1)']))
    print("Confusion Matrix (rows: true, cols: predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred)
    print(f"acc: {accuracy}, pre: {precision}, rec: {recall}, f1: {f1}")


# --- 主函数执行流程 ---
if __name__ == '__main__':
    try:
        df = load_and_prepare_data('../../spams_detection/datasets/crawler/LA/outputs/full_data_0617.csv')
        doc2vec_vectors, doc2vec_model, df = train_and_vectorize_doc2vec(df)
        df = calculate_suspicion_degree(df)
        feature_matrix = extract_other_features(df, doc2vec_vectors)
        X = feature_matrix.values
        y = df['label'].values
        print(f"\nFinal feature matrix shape: {X.shape}")
        train_and_evaluate(X, y)
    except (FileNotFoundError, ValueError) as e:
        print(e)