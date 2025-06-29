import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import numpy as np

# --- 1. 配置参数 ---
class Config:
    CSV_PATH = '../spams_dataset/LA/outputs/full_data_0617.csv'
    TEXT_COLUMN = 'content'
    LABEL_COLUMN = 'is_recommended'
    PRE_TRAINED_MODEL_NAME = 'roberta-base'
    MAX_LEN = 200
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    RANDOM_STATE = 42
    # 根据API文档，我们设定一个统一的步数
    LOG_EVAL_SAVE_STEPS = 500

# --- 2. 检查并设置设备 ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Using device: {device} ---")

# --- 3. 定义评估指标计算函数 ---
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

# --- 4. 主执行流程 ---
if __name__ == '__main__':
    # --- 数据加载与准备 ---
    print(f"Loading data from {Config.CSV_PATH}...")
    df = pd.read_csv(Config.CSV_PATH)
    df = df[[Config.TEXT_COLUMN, Config.LABEL_COLUMN]].dropna()
    df = df.rename(columns={Config.LABEL_COLUMN: 'label', Config.TEXT_COLUMN: 'text'})
    df['label'] = df['label'].astype(int)

    print(f"\nDataset Info: {len(df)} rows.")
    num_labels = df['label'].nunique()

    # 划分数据集
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=Config.RANDOM_STATE, stratify=df['label'])
    # 从测试集中划分出验证集
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=Config.RANDOM_STATE, stratify=df_test['label'])

    print(f"Dataset sizes: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    # --- Tokenization ---
    print("\nTokenizing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(Config.PRE_TRAINED_MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=Config.MAX_LEN)
    
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # --- 模型训练 ---
    print("\n--- Setting up model and trainer ---")
    model = RobertaForSequenceClassification.from_pretrained(Config.PRE_TRAINED_MODEL_NAME, num_labels=num_labels)
    
    # 【最终修正版，严格遵循 v4.52.3 文档】
    training_args = TrainingArguments(
        output_dir='./results_fakeroberta',
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=500,
        weight_decay=0.01,
        
        logging_dir='./logs_fakeroberta',
        logging_steps=Config.LOG_EVAL_SAVE_STEPS,
        
        # 开启评估
        do_eval=True,
        
        # 【此处是关键修正点】
        # 使用 'eval_steps' 而不是 'evaluation_steps'
        eval_steps=Config.LOG_EVAL_SAVE_STEPS,
        
        save_steps=Config.LOG_EVAL_SAVE_STEPS,
        save_total_limit=3,
        report_to="none" # 关闭 wandb/tensorboard 等报告
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 开始训练
    print("\n--- Starting Fine-tuning (fakeRoBERTa) ---")
    trainer.train()
    
    # --- 模型评估 ---
    print("\n--- Evaluating on the test set ---")
    test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    
    print("\n--- Final Baseline Results (fakeRoBERTa) ---")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1-score (Macro): {test_results['eval_f1']:.4f}")
    print(f"Test Precision (Macro): {test_results['eval_precision']:.4f}")
    print(f"Test Recall (Macro): {test_results['eval_recall']:.4f}")
    print(test_results)
    print("-------------------------------------------------")