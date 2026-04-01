"""
17-supervised_comparison.py
Supervised Model Comparison: Original Labels vs Re-evaluated Labels.

Purpose:
  Train 3 supervised classifiers on both the original and re-labeled
  training sets, then compare performance on the held-out test set.
  This demonstrates the impact of label quality on model performance.

Models:
  1. Decision Tree Classifier
  2. Logistic Regression (multinomial, saga solver)
  3. Ridge Classifier (linear regression adapted for classification)

Comparison:
  Each model is trained twice:
    - GROUP A: Original labels  → evaluate on test
    - GROUP B: Re-labeled data  → evaluate on test
  Plus XGBoost baseline (from training report) for context.

Outputs:
  - data/supervised_comparison_report.txt
  - plots/model_comparison_accuracy.png
  - plots/model_comparison_f1.png
  - plots/decision_tree_importance.png
  - plots/logistic_regression_coefficients.png
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
)

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

# Sampling for slow models (LogReg, Ridge) — None = use all
# Decision Tree can handle the full dataset
LOGREG_SAMPLE = 300_000     # Logistic Regression is slow with many classes
RIDGE_SAMPLE  = 300_000     # Ridge is faster but still benefits from sampling

# Model hyperparameters
DT_PARAMS = {
    'max_depth': 20,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'random_state': 42,
    'class_weight': 'balanced',
}

LR_PARAMS = {
    'solver': 'saga',
    'max_iter': 500,
    'random_state': 42,
    'n_jobs': -1,
    'C': 1.0,
    'tol': 1e-3,
}

RIDGE_PARAMS = {
    'alpha': 1.0,
    'class_weight': 'balanced',
}

RANDOM_STATE = 42

# Visual style
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FAFAFA',
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
})


# ============================================================
# HELPERS
# ============================================================

def sample_data(X, y, n_samples, random_state=42):
    """Stratified random subsample of X, y."""
    if n_samples is None or len(X) <= n_samples:
        return X, y
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), n_samples, replace=False)
    return X[idx], y[idx]


def parse_xgboost_baseline(data_dir):
    """
    Parse the XGBoost training report to extract baseline metrics.
    Returns (accuracy, f1_macro, f1_weighted) or None if not found.
    """
    report_path = os.path.join(data_dir, 'training_report.txt')
    if not os.path.exists(report_path):
        return None
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    acc = f1m = f1w = None
    # Look for patterns like "Test Accuracy:     0.3002%"
    m = re.search(r'Test Accuracy:\s+([\d.]+)', content)
    if m:
        acc = float(m.group(1))
    m = re.search(r'Test F1 \(macro\):\s+([\d.]+)', content)
    if m:
        f1m = float(m.group(1))
    m = re.search(r'Test F1 \(weighted\):\s*([\d.]+)', content)
    if m:
        f1w = float(m.group(1))

    if acc is not None:
        return {'accuracy': acc, 'f1_macro': f1m, 'f1_weighted': f1w}
    return None


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    report = []

    def log(msg):
        print(msg)
        report.append(msg)

    log("=" * 65)
    log("17-SUPERVISED_COMPARISON.PY — Original vs Re-labeled Labels")
    log("=" * 65)

    # ----------------------------------------------------------
    # STEP 1: Load Data
    # ----------------------------------------------------------
    log("\n[STEP 1] Loading data...")

    X_train_df = pd.read_parquet(os.path.join(DATA_DIR, 'X_train.parquet'))
    y_train_orig = pd.read_parquet(
        os.path.join(DATA_DIR, 'y_train.parquet')
    ).iloc[:, 0].values
    X_test_df = pd.read_parquet(os.path.join(DATA_DIR, 'X_test.parquet'))
    y_test = pd.read_parquet(
        os.path.join(DATA_DIR, 'y_test.parquet')
    ).iloc[:, 0].values

    # Handle NaN values (latitude/longitude have NaNs)
    nan_train = X_train_df.isnull().sum().sum()
    nan_test = X_test_df.isnull().sum().sum()
    if nan_train > 0 or nan_test > 0:
        log(f"  NaN detected: {nan_train} in train, {nan_test} in test — imputing with median")
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train_df)
        X_test = imputer.transform(X_test_df)
    else:
        X_train = X_train_df.values
        X_test = X_test_df.values

    with open(os.path.join(DATA_DIR, 'preprocessing_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)

    target_classes = np.array(artifacts['target_classes'])
    feature_names = artifacts['feature_names']
    n_classes = len(target_classes)

    log(f"  X_train: {X_train.shape}")
    log(f"  X_test:  {X_test.shape}")
    log(f"  Classes: {n_classes}")

    # Load re-labeled y_train
    relabeled_path = os.path.join(DATA_DIR, 'y_train_relabeled.parquet')
    if not os.path.exists(relabeled_path):
        log(f"\n  ERROR: {relabeled_path} not found!")
        log("  Run 16-unsupervised_clustering.py first.")
        return

    y_train_relabeled = pd.read_parquet(relabeled_path).iloc[:, 0].values
    n_changed = (y_train_orig != y_train_relabeled).sum()
    log(f"  Re-labeled training labels loaded ({n_changed:,} labels changed)")

    # Load XGBoost baseline
    xgb_baseline = parse_xgboost_baseline(DATA_DIR)
    if xgb_baseline:
        log(f"  XGBoost baseline: Acc={xgb_baseline['accuracy']:.4f}, "
            f"F1m={xgb_baseline['f1_macro']:.4f}")

    # ----------------------------------------------------------
    # STEP 2: Define Models
    # ----------------------------------------------------------
    log("\n[STEP 2] Defining models and training configurations...")

    models = {
        'Decision Tree': {
            'class': DecisionTreeClassifier,
            'params': DT_PARAMS,
            'sample': None,  # Use full data
        },
        'Logistic Regression': {
            'class': LogisticRegression,
            'params': LR_PARAMS,
            'sample': LOGREG_SAMPLE,
        },
        'Ridge Classifier': {
            'class': RidgeClassifier,
            'params': RIDGE_PARAMS,
            'sample': RIDGE_SAMPLE,
        },
    }

    for name, cfg in models.items():
        sample_str = (f"{cfg['sample']:,}" if cfg['sample']
                      else f"{len(X_train):,} (full)")
        log(f"  {name}: sample={sample_str}")
        for k, v in cfg['params'].items():
            log(f"    {k}: {v}")

    # ----------------------------------------------------------
    # STEP 3: Train & Evaluate All Models on Both Datasets
    # ----------------------------------------------------------
    log("\n[STEP 3] Training and evaluating models...")

    datasets = {
        'Original': y_train_orig,
        'Re-labeled': y_train_relabeled,
    }

    all_results = []  # [{model, dataset, accuracy, f1_macro, f1_weighted, time}]

    for model_name, cfg in models.items():
        for ds_name, y_train in datasets.items():
            log(f"\n  --- {model_name} on {ds_name} labels ---")

            # Subsample if needed
            X_tr, y_tr = sample_data(
                X_train, y_train,
                cfg['sample'],
                random_state=RANDOM_STATE,
            )
            log(f"  Training on {len(X_tr):,} samples...")

            # Train
            t0 = time.time()
            model = cfg['class'](**cfg['params'])
            model.fit(X_tr, y_tr)
            train_time = time.time() - t0
            log(f"  Training completed in {train_time:.1f}s")

            # Predict on test set
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            log(f"  Results:")
            log(f"    Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
            log(f"    F1 (macro):    {f1m:.4f}")
            log(f"    F1 (weighted): {f1w:.4f}")

            all_results.append({
                'Model': model_name,
                'Dataset': ds_name,
                'Accuracy': acc,
                'F1 Macro': f1m,
                'F1 Weighted': f1w,
                'Time (s)': train_time,
            })

            # Save the Decision Tree trained on original for importance plot
            if model_name == 'Decision Tree' and ds_name == 'Original':
                dt_model_orig = model

    # Add XGBoost baseline
    if xgb_baseline:
        all_results.append({
            'Model': 'XGBoost (baseline)',
            'Dataset': 'Original',
            'Accuracy': xgb_baseline['accuracy'],
            'F1 Macro': xgb_baseline['f1_macro'],
            'F1 Weighted': xgb_baseline['f1_weighted'],
            'Time (s)': 0,
        })

    df_results = pd.DataFrame(all_results)

    # ----------------------------------------------------------
    # STEP 4: Comparison Table
    # ----------------------------------------------------------
    log("\n" + "=" * 65)
    log("COMPARISON TABLE — All Models × Both Datasets")
    log("=" * 65)

    log(f"\n  {'Model':<25} {'Dataset':<12} {'Accuracy':>10} "
        f"{'F1 Macro':>10} {'F1 Weighted':>12} {'Time':>8}")
    log(f"  {'-'*77}")

    for _, row in df_results.iterrows():
        log(f"  {row['Model']:<25} {row['Dataset']:<12} "
            f"{row['Accuracy']:>10.4f} {row['F1 Macro']:>10.4f} "
            f"{row['F1 Weighted']:>12.4f} {row['Time (s)']:>7.1f}s")

    # Compute improvement for each model
    log(f"\n  IMPROVEMENT ANALYSIS (Re-labeled vs Original):")
    log(f"  {'Model':<25} {'Acc Δ':>10} {'F1m Δ':>10} {'F1w Δ':>10}")
    log(f"  {'-'*55}")

    for model_name in models.keys():
        orig = df_results[
            (df_results['Model'] == model_name) &
            (df_results['Dataset'] == 'Original')
        ]
        relab = df_results[
            (df_results['Model'] == model_name) &
            (df_results['Dataset'] == 'Re-labeled')
        ]

        if len(orig) > 0 and len(relab) > 0:
            acc_delta = relab.iloc[0]['Accuracy'] - orig.iloc[0]['Accuracy']
            f1m_delta = relab.iloc[0]['F1 Macro'] - orig.iloc[0]['F1 Macro']
            f1w_delta = relab.iloc[0]['F1 Weighted'] - orig.iloc[0]['F1 Weighted']

            sign_a = '+' if acc_delta >= 0 else ''
            sign_m = '+' if f1m_delta >= 0 else ''
            sign_w = '+' if f1w_delta >= 0 else ''

            log(f"  {model_name:<25} {sign_a}{acc_delta:>9.4f} "
                f"{sign_m}{f1m_delta:>9.4f} {sign_w}{f1w_delta:>9.4f}")

    # ----------------------------------------------------------
    # STEP 5: Comparison Plots
    # ----------------------------------------------------------
    log("\n[STEP 5] Generating comparison plots...")

    # --- 5a. Accuracy Comparison ---
    fig, ax = plt.subplots(figsize=(12, 7))

    model_names = list(models.keys())
    if xgb_baseline:
        model_names.append('XGBoost (baseline)')

    x = np.arange(len(model_names))
    width = 0.35

    # Original dataset values
    acc_orig = []
    acc_relab = []
    for mn in model_names:
        orig = df_results[
            (df_results['Model'] == mn) & (df_results['Dataset'] == 'Original')
        ]
        relab = df_results[
            (df_results['Model'] == mn) & (df_results['Dataset'] == 'Re-labeled')
        ]
        acc_orig.append(orig.iloc[0]['Accuracy'] if len(orig) > 0 else 0)
        acc_relab.append(relab.iloc[0]['Accuracy'] if len(relab) > 0 else 0)

    bars1 = ax.bar(x - width/2, acc_orig, width, label='Original Labels',
                   color='#3498db', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, acc_relab, width, label='Re-labeled',
                   color='#e74c3c', alpha=0.85, edgecolor='white')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy: Original vs Re-labeled Labels',
                 fontweight='bold', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_accuracy.png'),
                dpi=150)
    plt.close()
    log("  Saved: model_comparison_accuracy.png")

    # --- 5b. F1 Score Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, (metric, title) in enumerate([
        ('F1 Macro', 'F1 Score (Macro)'),
        ('F1 Weighted', 'F1 Score (Weighted)'),
    ]):
        ax = axes[i]

        vals_orig = []
        vals_relab = []
        for mn in model_names:
            orig = df_results[
                (df_results['Model'] == mn) &
                (df_results['Dataset'] == 'Original')
            ]
            relab = df_results[
                (df_results['Model'] == mn) &
                (df_results['Dataset'] == 'Re-labeled')
            ]
            vals_orig.append(
                orig.iloc[0][metric] if len(orig) > 0 else 0
            )
            vals_relab.append(
                relab.iloc[0][metric] if len(relab) > 0 else 0
            )

        b1 = ax.bar(x - width/2, vals_orig, width,
                     label='Original', color='#3498db', alpha=0.85)
        b2 = ax.bar(x + width/2, vals_relab, width,
                     label='Re-labeled', color='#e74c3c', alpha=0.85)

        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'{title}: Original vs Re-labeled',
                     fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bar in b1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in b2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_f1.png'), dpi=150)
    plt.close()
    log("  Saved: model_comparison_f1.png")

    # --- 5c. Decision Tree Feature Importance ---
    log("\n[STEP 5c] Decision Tree feature importance...")

    if dt_model_orig is not None:
        importances = dt_model_orig.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
        }).sort_values('importance', ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
        ax.barh(feat_imp['feature'], feat_imp['importance'], color=colors)
        ax.set_xlabel('Feature Importance (Gini)')
        ax.set_title('Top 15 Features — Decision Tree Classifier',
                      fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, 'decision_tree_importance.png'),
            dpi=150,
        )
        plt.close()
        log("  Saved: decision_tree_importance.png")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    log("\n" + "=" * 65)
    log("SUPERVISED COMPARISON COMPLETE — SUMMARY")
    log("=" * 65)
    log(f"  Models evaluated:      3 + XGBoost baseline")
    log(f"  Datasets:              Original labels / Re-labeled")
    log(f"  Training labels changed: {n_changed:,} "
        f"({n_changed/len(y_train_orig)*100:.2f}%)")
    log(f"  Test samples:          {len(X_test):,}")
    log(f"  Test labels:           Original (unchanged)")
    log(f"\n  Plots generated:")
    log(f"  - model_comparison_accuracy.png")
    log(f"  - model_comparison_f1.png")
    log(f"  - decision_tree_importance.png")
    log("=" * 65)

    # Save report
    report_path = os.path.join(DATA_DIR, 'supervised_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    log(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
