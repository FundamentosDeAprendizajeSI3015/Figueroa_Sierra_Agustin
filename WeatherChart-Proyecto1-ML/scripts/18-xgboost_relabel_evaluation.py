"""
18-xgboost_relabel_evaluation.py
Train XGBoost Classifier specifically on the new Relabeled Trainingset
and evaluate it to measure the ultimate performance gain given by 
the Unsupervised Clustering Label Cleanup.

Outputs:
  - Plots generated in General/Presentacion_WeatherChart/plots/
  - data/xgboost_relabel_report.txt
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
import re

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'General', 'Presentacion_WeatherChart', 'plots')

MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',              
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 15,        
    'random_state': 42,
    'n_jobs': -1,                       
    'verbosity': 1,
}

VAL_SIZE = 0.2   
RANDOM_STATE = 42

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FAFAFA',
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
})

def parse_xgboost_baseline(data_dir):
    """Parse original XGBoost baseline metrics."""
    report_path = os.path.join(data_dir, 'training_report.txt')
    if not os.path.exists(report_path):
        return {'accuracy': 0.3002, 'f1_macro': 0.0845} # Known baseline
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    acc = f1m = f1w = None
    m = re.search(r'Test Accuracy:\s+([\d.]+)', content)
    if m: acc = float(m.group(1))
    m = re.search(r'Test F1 \(macro\):\s+([\d.]+)', content)
    if m: f1m = float(m.group(1))
    m = re.search(r'Test F1 \(weighted\):\s*([\d.]+)', content)
    if m: f1w = float(m.group(1))

    if acc is not None:
        return {'accuracy': acc, 'f1_macro': f1m, 'f1_weighted': f1w}
    return {'accuracy': 0.3002, 'f1_macro': 0.0845, 'f1_weighted': 0.2216}

def main():
    report = []
    def log(msg):
        print(msg)
        report.append(msg)

    log("=" * 65)
    log("18-XGBOOST_RELABEL_EVALUATION.PY")
    log("Training XGBoost on Relabeled Data")
    log("=" * 65)

    # 1. Load Data
    log("\n[STEP 1] Loading Data...")
    X_train_df = pd.read_parquet(os.path.join(DATA_DIR, 'X_train.parquet'))
    relabeled_path = os.path.join(DATA_DIR, 'y_train_relabeled.parquet')
    if not os.path.exists(relabeled_path):
        log("ERROR: y_train_relabeled.parquet not found. Run clustering first.")
        return
    y_train = pd.read_parquet(relabeled_path).iloc[:, 0].values
    
    X_test_df = pd.read_parquet(os.path.join(DATA_DIR, 'X_test.parquet'))
    y_test = pd.read_parquet(os.path.join(DATA_DIR, 'y_test.parquet')).iloc[:, 0].values

    # NaNs Handling
    nan_train = X_train_df.isnull().sum().sum()
    if nan_train > 0:
        log("  Imputing NaNs with median...")
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train_df)
        X_test = imputer.transform(X_test_df)
    else:
        X_train = X_train_df.values
        X_test = X_test_df.values

    log(f"  Training shape: {X_train.shape}")
    log(f"  Testing shape:  {X_test.shape}")

    xgb_baseline = parse_xgboost_baseline(DATA_DIR)
    log(f"  XGBoost Original Baseline -> Acc: {xgb_baseline['accuracy']:.4f}")

    # 2. Train-Val split for Early Stopping
    log("\n[STEP 2] Splitting validation set (20%) for early stopping...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, 
        random_state=RANDOM_STATE, stratify=y_train
    )

    # 3. Handle non-continuous labels for XGBoost
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_tr_encoded = le.fit_transform(y_tr)
    y_val_encoded = le.transform(y_val)

    log(f"\n[STEP 3] Training XGBoost Classifier on {len(le.classes_)} distinct survived classes...")
    model = XGBClassifier(**MODEL_PARAMS)
    
    t0 = time.time()
    model.fit(
        X_tr, y_tr_encoded,
        eval_set=[(X_val, y_val_encoded)],
        verbose=False
    )
    train_time = time.time() - t0
    log(f"  Training completed in {train_time:.1f}s")
    log(f"  Best iteration: {model.best_iteration}")

    # 4. Evaluate
    log("\n[STEP 4] Evaluating on untouched TEST Set (Original Labels)...")
    y_pred_encoded = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded) # Map back to true genre IDs
    
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    log(f"  Relabeled XGBoost Results:")
    log(f"    Accuracy:   {acc:.4f} ({acc*100:.2f}%)")
    log(f"    F1 (macro): {f1m:.4f}")
    
    acc_diff = acc - xgb_baseline['accuracy']
    
    log(f"\n  === ULTIMATE COMPARISON ===")
    log(f"  Original XGBoost Acc:  {xgb_baseline['accuracy']:.4f}")
    log(f"  Relabeled XGBoost Acc: {acc:.4f}")
    log(f"  Net Delta (Improvement): {'+' if acc_diff>0 else ''}{acc_diff:.4f}")

    # 5. Plot Comparison
    log("\n[STEP 5] Generating Comparison Plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['XGBoost\n(Original Labels)', 'XGBoost\n(Re-labeled Data)'], 
                  [xgb_baseline['accuracy'], acc],
                  color=['#bdc3c7', '#27ae60'], width=0.5)
                  
    ax.set_ylabel('Test Accuracy')
    ax.set_title('The Impact of Label Quality: XGBoost Classifier', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(acc, xgb_baseline['accuracy']) + 0.1)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.4f} ({h*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'xgboost_ultimate_comparison.png'), dpi=150)
    plt.close()
    log("  Saved: xgboost_ultimate_comparison.png")

    report_path = os.path.join(DATA_DIR, 'xgboost_relabel_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\\n'.join(report))
    log(f"Report saved to {report_path}")

if __name__ == '__main__':
    main()
