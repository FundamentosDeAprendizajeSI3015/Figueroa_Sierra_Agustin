"""
14-train_model.py
Professional XGBoost training pipeline for Music Genre prediction.

Data Leakage Prevention Strategy:
  - The EXISTING X_train/y_train (from 13-preprocess.py) is sub-split into:
      * train_sub (80%)  → used for model.fit()
      * validation (20%) → used ONLY for early stopping monitoring
  - The EXISTING X_test/y_test remains COMPLETELY UNTOUCHED until final evaluation.
  - Early stopping decisions use ONLY the validation set.
  - Final metrics are computed ONLY on the test set.

Outputs:
  - data/xgboost_model.json         (trained model artifact)
  - data/training_report.txt        (full metrics report)
  - plots/confusion_matrix.png      (top-20 genres heatmap)
  - plots/feature_importance.png    (top-15 features by gain)
  - plots/learning_curves.png       (train vs validation loss)
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',              # Fast histogram-based method
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 15,        # Stop if val loss doesn't improve for 15 rounds
    'random_state': 42,
    'n_jobs': -1,                       # Use all CPU cores
    'verbosity': 1,
}

# Early stopping patience is set in MODEL_PARAMS above

# Validation split (from existing train set)
VAL_SIZE = 0.2   # 20% of train → validation
VAL_RANDOM_STATE = 42


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    report_lines = []

    def log(msg):
        """Print and store for report."""
        print(msg)
        report_lines.append(msg)

    log("=" * 60)
    log("14-TRAIN_MODEL.PY — XGBoost Training Pipeline")
    log("=" * 60)

    # ----------------------------------------------------------
    # STEP 1: Load Preprocessed Data
    # ----------------------------------------------------------
    log("\n[STEP 1] Loading preprocessed data...")

    X_train_full = pd.read_parquet(os.path.join(DATA_DIR, 'X_train.parquet'))
    y_train_full = pd.read_parquet(os.path.join(DATA_DIR, 'y_train.parquet')).iloc[:, 0]
    X_test = pd.read_parquet(os.path.join(DATA_DIR, 'X_test.parquet'))
    y_test = pd.read_parquet(os.path.join(DATA_DIR, 'y_test.parquet')).iloc[:, 0]

    log(f"  X_train_full: {X_train_full.shape}")
    log(f"  X_test:       {X_test.shape} (HELD OUT — NOT used during training)")
    log(f"  Classes:      {y_train_full.nunique()}")

    # Load preprocessing artifacts for label decoding
    with open(os.path.join(DATA_DIR, 'preprocessing_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)

    target_classes = artifacts['target_classes']
    feature_names = artifacts['feature_names']
    log(f"  Features:     {len(feature_names)}")

    # ----------------------------------------------------------
    # STEP 2: Create Validation Split (from train, NO test leakage)
    # ----------------------------------------------------------
    log("\n[STEP 2] Creating validation split from training data...")
    log(f"  Strategy: {100*(1-VAL_SIZE):.0f}% train / {100*VAL_SIZE:.0f}% validation")
    log(f"  Test set is COMPLETELY ISOLATED — zero leakage guaranteed.")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE,
        random_state=VAL_RANDOM_STATE,
        stratify=y_train_full,
    )

    log(f"  X_train (fit):      {X_train.shape}")
    log(f"  X_val (monitoring): {X_val.shape}")
    log(f"  X_test (final):     {X_test.shape}")

    # Free memory
    del X_train_full, y_train_full

    # ----------------------------------------------------------
    # STEP 3: Train XGBoost
    # ----------------------------------------------------------
    log("\n[STEP 3] Training XGBoost Classifier...")
    log(f"  Hyperparameters:")
    for k, v in MODEL_PARAMS.items():
        log(f"    {k}: {v}")
    log(f"  Early stopping patience: {MODEL_PARAMS['early_stopping_rounds']} rounds")

    model = XGBClassifier(**MODEL_PARAMS)

    start_time = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,  # Print every 50 rounds
    )

    elapsed = time.time() - start_time
    best_iter = getattr(model, 'best_iteration', MODEL_PARAMS['n_estimators'])
    best_score = getattr(model, 'best_score', None)
    log(f"\n  Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    log(f"  Best iteration: {best_iter}")
    if best_score is not None:
        log(f"  Best validation mlogloss: {best_score:.6f}")

    # ----------------------------------------------------------
    # STEP 4: Evaluate on TEST SET (first and only time)
    # ----------------------------------------------------------
    log("\n[STEP 4] Evaluating on HELD-OUT test set...")
    log("  (This is the FIRST time the model sees this data)")

    y_pred = model.predict(X_test)

    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    log(f"\n  ┌─────────────────────────────────────┐")
    log(f"  │  TEST SET RESULTS (No Leakage)       │")
    log(f"  ├─────────────────────────────────────┤")
    log(f"  │  Accuracy:        {accuracy:>8.4f}  ({accuracy*100:.2f}%) │")
    log(f"  │  F1 (macro):      {f1_macro:>8.4f}           │")
    log(f"  │  F1 (weighted):   {f1_weighted:>8.4f}           │")
    log(f"  │  Random baseline: {1/len(target_classes):>8.4f}  ({100/len(target_classes):.2f}%) │")
    log(f"  └─────────────────────────────────────┘")

    # Full classification report
    report = classification_report(
        y_test, y_pred,
        target_names=target_classes,
        zero_division=0,
    )
    log(f"\n  Classification Report (per genre):")
    log(report)

    # ----------------------------------------------------------
    # STEP 5: Confusion Matrix (Top 20 Genres)
    # ----------------------------------------------------------
    log("\n[STEP 5] Generating confusion matrix (top 20 genres)...")

    # Get top 20 most frequent genres in test set
    genre_counts = pd.Series(y_test).value_counts()
    top_20_indices = genre_counts.head(20).index.tolist()
    top_20_names = [target_classes[i] for i in top_20_indices]

    # Filter predictions to top 20
    mask = np.isin(y_test, top_20_indices)
    y_test_top = y_test[mask]
    y_pred_top = y_pred[mask]

    cm = confusion_matrix(y_test_top, y_pred_top, labels=top_20_indices)
    # Normalize by true label (row-wise)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=top_20_names,
        yticklabels=top_20_names,
        ax=ax,
        vmin=0, vmax=1,
        linewidths=0.5,
    )
    ax.set_xlabel('Predicted Genre', fontsize=12)
    ax.set_ylabel('True Genre', fontsize=12)
    ax.set_title('Confusion Matrix — Top 20 Genres (Normalized by Row)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log(f"  Saved: {cm_path}")

    # ----------------------------------------------------------
    # STEP 6: Feature Importance (Top 15)
    # ----------------------------------------------------------
    log("\n[STEP 6] Generating feature importance plot...")

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
    ax.barh(feat_imp['feature'], feat_imp['importance'], color=colors)
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title('Top 15 Feature Importance — XGBoost', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    fi_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(fi_path, dpi=150)
    plt.close()
    log(f"  Saved: {fi_path}")

    # ----------------------------------------------------------
    # STEP 7: Learning Curves (train vs validation loss)
    # ----------------------------------------------------------
    log("\n[STEP 7] Generating learning curves...")

    results = model.evals_result()
    train_loss = results['validation_0']['mlogloss']
    val_loss = results['validation_1']['mlogloss']

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label='Train Loss', linewidth=2, color='#2196F3')
    ax.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='#FF5722')
    ax.axvline(x=best_iter, color='green', linestyle='--',
               alpha=0.7, label=f'Best Iteration ({best_iter})')
    ax.set_xlabel('Boosting Round', fontsize=12)
    ax.set_ylabel('Multi-class Log Loss', fontsize=12)
    ax.set_title('Learning Curves — Train vs Validation', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    lc_path = os.path.join(PLOTS_DIR, 'learning_curves.png')
    plt.savefig(lc_path, dpi=150)
    plt.close()
    log(f"  Saved: {lc_path}")

    # ----------------------------------------------------------
    # STEP 8: Save Model & Report
    # ----------------------------------------------------------
    log("\n[STEP 8] Saving model and report...")

    # Save model
    model_path = os.path.join(DATA_DIR, 'xgboost_model.json')
    model.save_model(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    log(f"  Model saved: {model_path} ({model_size:.1f} MB)")

    # Final summary
    log("\n" + "=" * 60)
    log("TRAINING COMPLETE — SUMMARY")
    log("=" * 60)
    log(f"  Model:             XGBoost Classifier")
    log(f"  Train samples:     {len(X_train):>12,}")
    log(f"  Validation samples:{len(X_val):>12,}")
    log(f"  Test samples:      {len(X_test):>12,}")
    log(f"  Features:          {X_train.shape[1]:>12}")
    log(f"  Target classes:    {len(target_classes):>12}")
    log(f"  Best iteration:    {best_iter:>12}")
    log(f"  Test Accuracy:     {accuracy:>11.4f}%")
    log(f"  Test F1 (macro):   {f1_macro:>11.4f}")
    log(f"  Test F1 (weighted):{f1_weighted:>11.4f}")
    log(f"  Data Leakage:      {'NONE':>12}")
    log(f"  Training time:     {elapsed:>10.1f}s")
    log("=" * 60)

    # Save report
    report_path = os.path.join(DATA_DIR, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
