"""
FIRE-UdeA v2.0 — Métricas Detalladas y Visualizaciones
=======================================================
Lee los resultados del pipeline y genera:
  - Métricas detalladas por modelo y por fold
  - Confusion matrices
  - ROC y PR curves por fold
  - Per-fold AUC breakdown
  - Threshold analysis
  - Probability distribution (predicted probabilities)
  - Prediction heatmap (unit × year)
  - Radar chart comparativo
  - Summary dashboard

Salida: results/metrics_viz/ con PNG y CSV.

Uso:
    python FIRE_UdeA_metrics.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    matthews_corrcoef, classification_report
)

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
VIZ_DIR = os.path.join(RESULTS_DIR, 'final_metrics')
os.makedirs(VIZ_DIR, exist_ok=True)

ANOMALOUS_YEARS = {2019, 2020}
MODEL_ORDER = [
    '1_LogisticRegression_L2',
    '2_ElasticNet_Logistic',
    '3_DecisionTree',
    '4_RandomForest',
    '5_LightGBM',
]
MODEL_SHORT_NAMES = {
    '1_LogisticRegression_L2': 'LogReg L2',
    '2_ElasticNet_Logistic': 'ElasticNet',
    '3_DecisionTree': 'Decision Tree',
    '4_RandomForest': 'Random Forest',
    '5_LightGBM': 'LightGBM',
}
PALETTE = {
    'LogReg L2': '#2196F3',
    'ElasticNet': '#4CAF50',
    'Decision Tree': '#FF9800',
    'Random Forest': '#F44336',
    'LightGBM': '#9C27B0',
}

sns.set_theme(style='whitegrid', font_scale=1.05)


def load_predictions():
    """Load all prediction files."""
    preds = {}
    for model in MODEL_ORDER:
        path = os.path.join(RESULTS_DIR, f'predictions_{model}.csv')
        if os.path.exists(path):
            preds[model] = pd.read_csv(path)
    return preds


def load_metrics():
    """Load LOYO metrics."""
    path = os.path.join(RESULTS_DIR, 'metrics_loyo.csv')
    return pd.read_csv(path)


# ============================================================
# 1. COMPREHENSIVE METRICS TABLE
# ============================================================
def compute_full_metrics(all_preds):
    """Compute comprehensive metrics for each model."""
    rows = []
    for model_name, df in all_preds.items():
        y_true = df['y_true'].values
        y_prob = df['y_prob'].values
        y_pred = df['y_pred'].values
        short = MODEL_SHORT_NAMES[model_name]

        n = len(y_true)
        n_pos = y_true.sum()
        n_neg = n - n_pos

        # Core metrics
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Stable folds (exclude anomalous years)
        mask_stable = ~df['anio'].isin(ANOMALOUS_YEARS)
        if mask_stable.sum() > 0:
            y_t_s = df.loc[mask_stable, 'y_true'].values
            y_p_s = df.loc[mask_stable, 'y_prob'].values
            if len(np.unique(y_t_s)) >= 2:
                roc_auc_stable = roc_auc_score(y_t_s, y_p_s)
            else:
                roc_auc_stable = np.nan
        else:
            roc_auc_stable = np.nan

        rows.append({
            'Model': short,
            'N': n,
            'ROC-AUC': roc_auc,
            'ROC-AUC (stable)': roc_auc_stable,
            'PR-AUC': pr_auc,
            'Brier': brier,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'Specificity': specificity,
            'F1': f1,
            'MCC': mcc,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        })

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(VIZ_DIR, 'full_metrics.csv'), index=False)
    print("\n=== COMPREHENSIVE METRICS ===")
    print(result[['Model', 'ROC-AUC', 'ROC-AUC (stable)', 'PR-AUC', 'Brier', 'Accuracy', 'F1', 'MCC']].to_string(index=False))
    return result


# ============================================================
# 2. CONFUSION MATRICES PER MODEL
# ============================================================
def plot_confusion_matrices(all_preds):
    """Plot confusion matrix for each model in a single figure."""
    models = [m for m in MODEL_ORDER if m in all_preds]
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for i, model_name in enumerate(models):
        df = all_preds[model_name]
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        short = MODEL_SHORT_NAMES[model_name]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_pct = cm.astype(float) / cm.sum() * 100

        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Sano (0)', 'Riesgo (1)'],
                    yticklabels=['Sano (0)', 'Riesgo (1)'],
                    linewidths=1, linecolor='white',
                    annot_kws={'size': 14, 'fontweight': 'bold'})

        # Overlay percentages
        for r in range(2):
            for c in range(2):
                ax.text(c + 0.5, r + 0.7, f"({cm_pct[r, c]:.1f}%)",
                        ha='center', va='center', fontsize=9, color='gray')

        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        auc = roc_auc_score(df['y_true'], df['y_prob'])
        ax.set_title(f'{short}\nAUC={auc:.3f}', fontsize=12, fontweight='bold')

    plt.suptitle('FIRE-UdeA — Confusion Matrices (All LOYO Predictions)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '01_confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_confusion_matrices.png")


# ============================================================
# 3. ROC CURVES (ALL MODELS + PER FOLD FOR BEST)
# ============================================================
def plot_roc_curves(all_preds, metrics_df):
    """ROC curves: all models overlay + per-fold for best model."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: all models
    ax = axes[0]
    for model_name in MODEL_ORDER:
        if model_name not in all_preds:
            continue
        df = all_preds[model_name]
        y_true = df['y_true'].values
        y_prob = df['y_prob'].values
        short = MODEL_SHORT_NAMES[model_name]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=PALETTE[short], linewidth=2.2,
                label=f'{short} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.03, color='gray')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Right: per fold for best model (LogReg)
    ax = axes[1]
    best_model = '1_LogisticRegression_L2'
    if best_model in all_preds:
        df = all_preds[best_model]
        test_metrics = metrics_df[(metrics_df['model'] == best_model) & (metrics_df['split'] == 'test')]

        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(test_metrics)))
        for idx, (_, row) in enumerate(test_metrics.iterrows()):
            year = int(row['year'])
            fold_df = df[df['anio'] == year]
            if len(fold_df) < 2 or len(fold_df['y_true'].unique()) < 2:
                continue
            fpr, tpr, _ = roc_curve(fold_df['y_true'], fold_df['y_prob'])
            auc = roc_auc_score(fold_df['y_true'], fold_df['y_prob'])
            marker = '⚠' if year in ANOMALOUS_YEARS else ''
            ax.plot(fpr, tpr, color=cmap[idx], linewidth=1.8,
                    label=f'{year}{marker} (AUC={auc:.2f})', alpha=0.85)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('LogReg L2 — ROC per LOYO Fold', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '02_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_roc_curves.png")


# ============================================================
# 4. PRECISION-RECALL CURVES
# ============================================================
def plot_pr_curves(all_preds):
    """Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for model_name in MODEL_ORDER:
        if model_name not in all_preds:
            continue
        df = all_preds[model_name]
        y_true = df['y_true'].values
        y_prob = df['y_prob'].values
        short = MODEL_SHORT_NAMES[model_name]

        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(rec_curve, prec_curve, color=PALETTE[short], linewidth=2.2,
                label=f'{short} (AP={pr_auc:.3f})')

    baseline = all_preds[MODEL_ORDER[0]]['y_true'].mean()
    ax.axhline(baseline, color='gray', linestyle='--', alpha=0.4, label=f'Baseline ({baseline:.2f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('FIRE-UdeA — Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '03_pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_pr_curves.png")


# ============================================================
# 5. PER-FOLD AUC BREAKDOWN (BAR CHART)
# ============================================================
def plot_per_fold_auc(metrics_df, all_preds):
    """Bar chart showing AUC per year per model."""
    test_df = metrics_df[metrics_df['split'] == 'test'].copy()
    test_df['short_model'] = test_df['model'].map(MODEL_SHORT_NAMES)

    fig, ax = plt.subplots(figsize=(14, 7))

    models_in_data = [m for m in MODEL_ORDER if m in all_preds]
    n_models = len(models_in_data)
    years = sorted(test_df['year'].unique())
    bar_width = 0.15
    x = np.arange(len(years))

    for i, model in enumerate(models_in_data):
        short = MODEL_SHORT_NAMES[model]
        model_data = test_df[test_df['model'] == model].set_index('year')
        aucs = [model_data.loc[y, 'roc_auc'] if y in model_data.index else np.nan for y in years]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, aucs, bar_width * 0.9, label=short,
                      color=PALETTE[short], alpha=0.85, edgecolor='white', linewidth=0.8)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='Random (0.5)')
    ax.axhline(0.55, color='#FF9800', linestyle=':', alpha=0.5, label='Target (0.55)')

    # Mark anomalous years
    for i, y in enumerate(years):
        if y in ANOMALOUS_YEARS:
            ax.axvspan(i - 0.45, i + 0.45, alpha=0.08, color='red')
            ax.text(i, -0.08, '⚠', ha='center', fontsize=12, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11)
    ax.set_xlabel('Test Year (LOYO Fold)', fontsize=12)
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title('FIRE-UdeA — Per-Fold AUC Breakdown by Model', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    ax.set_ylim(-0.1, 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '04_per_fold_auc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_per_fold_auc.png")


# ============================================================
# 6. PROBABILITY DISTRIBUTIONS
# ============================================================
def plot_probability_distributions(all_preds):
    """Distribution of predicted probabilities by true label for best model."""
    best = '1_LogisticRegression_L2'
    if best not in all_preds:
        return
    df = all_preds[best]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: histogram
    ax = axes[0]
    for label_val in [0, 1]:
        data = df.loc[df['y_true'] == label_val, 'y_prob']
        color = '#4CAF50' if label_val == 0 else '#F44336'
        label_text = f'Sano (n={len(data)})' if label_val == 0 else f'Riesgo (n={len(data)})'
        ax.hist(data, bins=15, alpha=0.65, color=color, label=label_text, edgecolor='white')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold=0.5')
    ax.set_xlabel('Predicted Probability P(Risk)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('LogReg L2 — Probability Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # Right: box plot per year
    ax = axes[1]
    df_plot = df.copy()
    df_plot['label_str'] = df_plot['y_true'].map({0: 'Sano', 1: 'Riesgo'})
    sns.boxplot(data=df_plot, x='anio', y='y_prob', hue='label_str',
                palette={'Sano': '#4CAF50', 'Riesgo': '#F44336'},
                ax=ax, width=0.6)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Predicted P(Risk)', fontsize=12)
    ax.set_title('Probabilities by Year & True Label', fontsize=13, fontweight='bold')
    ax.legend(title='True Label', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '05_probability_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_probability_distributions.png")


# ============================================================
# 7. PREDICTION HEATMAP (UNIT × YEAR)
# ============================================================
def plot_prediction_heatmaps(all_preds):
    """Heatmap showing predictions vs actuals for best model."""
    best = '1_LogisticRegression_L2'
    if best not in all_preds:
        return
    df = all_preds[best]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1: True labels
    pivot_true = df.pivot_table(index='unidad', columns='anio', values='y_true', aggfunc='first')
    ax = axes[0]
    sns.heatmap(pivot_true, annot=True, fmt='.0f', cmap=['#4CAF50', '#F44336'],
                linewidths=1, linecolor='white', ax=ax, cbar=False, vmin=0, vmax=1)
    ax.set_title('True Labels', fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(labelsize=9)

    # 2: Predicted probabilities
    pivot_prob = df.pivot_table(index='unidad', columns='anio', values='y_prob', aggfunc='first')
    ax = axes[1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap_risk = LinearSegmentedColormap.from_list('risk', ['#4CAF50', '#FFEB3B', '#F44336'])
    sns.heatmap(pivot_prob, annot=True, fmt='.2f', cmap=cmap_risk,
                linewidths=1, linecolor='white', ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'P(Risk)', 'shrink': 0.8})
    ax.set_title('Predicted P(Risk) — LogReg L2', fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(labelsize=9)

    # 3: Errors (true ≠ predicted)
    df_err = df.copy()
    df_err['correct'] = (df_err['y_true'] == df_err['y_pred']).astype(int)
    pivot_err = df_err.pivot_table(index='unidad', columns='anio', values='correct', aggfunc='first')
    ax = axes[2]
    sns.heatmap(pivot_err, annot=True, fmt='.0f', cmap=['#F44336', '#4CAF50'],
                linewidths=1, linecolor='white', ax=ax, cbar=False, vmin=0, vmax=1)
    ax.set_title('Correct Predictions (1=✓)', fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(labelsize=9)

    plt.suptitle('FIRE-UdeA — Prediction Heatmaps (LogReg L2)',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '06_prediction_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_prediction_heatmaps.png")


# ============================================================
# 8. RADAR CHART — MODEL COMPARISON
# ============================================================
def plot_radar_chart(metrics_table):
    """Radar chart comparing all models across key metrics."""
    metrics_to_plot = ['ROC-AUC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    metrics_to_plot = [m for m in metrics_to_plot if m in metrics_table.columns]
    N = len(metrics_to_plot)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for _, row in metrics_table.iterrows():
        model = row['Model']
        values = [row[m] for m in metrics_to_plot]
        # Normalize MCC from [-1,1] to [0,1] for visual consistency
        mcc_idx = metrics_to_plot.index('MCC') if 'MCC' in metrics_to_plot else -1
        if mcc_idx >= 0:
            values[mcc_idx] = (values[mcc_idx] + 1) / 2  # Map [-1,1] → [0,1]
        values += values[:1]

        color = PALETTE.get(model, '#999999')
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model, markersize=5)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    labels = [m if m != 'MCC' else 'MCC (norm)' for m in metrics_to_plot]
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title('FIRE-UdeA — Model Performance Radar', fontsize=14,
                 fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '07_radar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_radar_chart.png")


# ============================================================
# 9. THRESHOLD ANALYSIS
# ============================================================
def plot_threshold_analysis(all_preds):
    """Show precision, recall, F1 as function of threshold for best model."""
    best = '1_LogisticRegression_L2'
    if best not in all_preds:
        return
    df = all_preds[best]
    y_true = df['y_true'].values
    y_prob = df['y_prob'].values

    thresholds = np.linspace(0.05, 0.95, 50)
    metrics_by_thresh = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        metrics_by_thresh.append({
            'threshold': t,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        })

    t_df = pd.DataFrame(metrics_by_thresh)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: precision, recall, F1 vs threshold
    ax = axes[0]
    ax.plot(t_df['threshold'], t_df['precision'], 'b-', linewidth=2, label='Precision')
    ax.plot(t_df['threshold'], t_df['recall'], 'r-', linewidth=2, label='Recall')
    ax.plot(t_df['threshold'], t_df['f1'], 'g-', linewidth=2.5, label='F1 Score')
    best_f1_idx = t_df['f1'].idxmax()
    best_t = t_df.loc[best_f1_idx, 'threshold']
    best_f1_val = t_df.loc[best_f1_idx, 'f1']
    ax.axvline(best_t, color='green', linestyle=':', alpha=0.5)
    ax.scatter([best_t], [best_f1_val], color='green', s=100, zorder=5,
               label=f'Best F1={best_f1_val:.3f} @ t={best_t:.2f}')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision / Recall / F1 vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)

    # Right: Accuracy + Specificity vs threshold
    ax = axes[1]
    ax.plot(t_df['threshold'], t_df['accuracy'], '#2196F3', linewidth=2, label='Accuracy')
    ax.plot(t_df['threshold'], t_df['specificity'], '#FF9800', linewidth=2, label='Specificity')
    ax.plot(t_df['threshold'], t_df['recall'], '#F44336', linewidth=2, label='Recall (Sensitivity)')
    # Youden's J optimal
    j_scores = t_df['recall'] + t_df['specificity'] - 1
    best_j_idx = j_scores.idxmax()
    best_j_t = t_df.loc[best_j_idx, 'threshold']
    ax.axvline(best_j_t, color='purple', linestyle=':', alpha=0.5,
               label=f"Youden's J optimal @ t={best_j_t:.2f}")
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Accuracy / Sensitivity / Specificity vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)

    plt.suptitle('FIRE-UdeA — Threshold Analysis (LogReg L2)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '08_threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_threshold_analysis.png")

    t_df.to_csv(os.path.join(VIZ_DIR, 'threshold_analysis.csv'), index=False)


# ============================================================
# 10. TRAIN vs TEST GAP ANALYSIS
# ============================================================
def plot_train_test_gap(metrics_df, all_preds):
    """Visualize overfitting gap per model."""
    models = [m for m in MODEL_ORDER if m in all_preds]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: bar chart of train vs test AUC
    ax = axes[0]
    train_means = []
    test_means = []
    short_names = []

    for model in models:
        short = MODEL_SHORT_NAMES[model]
        short_names.append(short)
        t_test = metrics_df[(metrics_df['model'] == model) & (metrics_df['split'] == 'test')]['roc_auc'].dropna()
        t_train = metrics_df[(metrics_df['model'] == model) & (metrics_df['split'] == 'train')]['roc_auc'].dropna()
        test_means.append(t_test.mean())
        train_means.append(t_train.mean())

    x = np.arange(len(models))
    ax.bar(x - 0.18, train_means, 0.35, label='Train', color='#42A5F5', alpha=0.8, edgecolor='white')
    ax.bar(x + 0.18, test_means, 0.35, label='Test', color='#F44336', alpha=0.8, edgecolor='white')

    for i in range(len(models)):
        gap = train_means[i] - test_means[i]
        color = '#F44336' if gap > 0.15 else '#4CAF50'
        ax.annotate(f'Δ={gap:.3f}', (i, max(train_means[i], test_means[i]) + 0.02),
                    ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel('Mean ROC-AUC', fontsize=12)
    ax.set_title('Train vs Test AUC — Overfitting Gap', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    # Right: gap progression (complexity vs gap)
    ax = axes[1]
    gaps = [t - s for t, s in zip(train_means, test_means)]
    colors = ['#4CAF50' if g <= 0.15 else '#F44336' for g in gaps]
    ax.bar(short_names, gaps, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axhline(0.15, color='#F44336', linestyle='--', alpha=0.5, label='Overfitting threshold (0.15)')
    ax.set_ylabel('AUC Gap (Train - Test)', fontsize=12)
    ax.set_title('Overfitting Gap by Model Complexity →', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', labelsize=10)

    for i, g in enumerate(gaps):
        ax.text(i, g + 0.005, f'{g:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '09_train_test_gap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_train_test_gap.png")


# ============================================================
# 11. PER-UNIT PERFORMANCE
# ============================================================
def plot_per_unit_performance(all_preds):
    """Performance breakdown by academic unit."""
    best = '1_LogisticRegression_L2'
    if best not in all_preds:
        return
    df = all_preds[best]

    unit_metrics = []
    for unit in sorted(df['unidad'].unique()):
        unit_df = df[df['unidad'] == unit]
        y_true = unit_df['y_true'].values
        y_pred = unit_df['y_pred'].values
        n = len(y_true)
        acc = accuracy_score(y_true, y_pred)
        correct = (y_true == y_pred).sum()
        prevalence = y_true.mean()
        unit_metrics.append({
            'Unidad': unit[:15],
            'N': n,
            'Prevalence': prevalence,
            'Accuracy': acc,
            'Correct': correct,
            'Errors': n - correct,
        })

    unit_df_result = pd.DataFrame(unit_metrics).sort_values('Accuracy', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: accuracy by unit
    ax = axes[0]
    colors = ['#4CAF50' if a >= 0.7 else '#FF9800' if a >= 0.5 else '#F44336'
              for a in unit_df_result['Accuracy']]
    ax.barh(unit_df_result['Unidad'], unit_df_result['Accuracy'],
            color=colors, alpha=0.85, edgecolor='white')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('LogReg L2 — Accuracy per Unit', fontsize=13, fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlim(0, 1.05)

    for i, (_, row) in enumerate(unit_df_result.iterrows()):
        ax.text(row['Accuracy'] + 0.02, i, f"{row['Correct']}/{row['N']}", va='center', fontsize=10)

    # Right: errors per unit
    ax = axes[1]
    ax.barh(unit_df_result['Unidad'], unit_df_result['Errors'],
            color='#F44336', alpha=0.7, edgecolor='white', label='Errors')
    ax.barh(unit_df_result['Unidad'], unit_df_result['Correct'],
            left=unit_df_result['Errors'],
            color='#4CAF50', alpha=0.7, edgecolor='white', label='Correct')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title('Correct vs Errors per Unit', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.suptitle('FIRE-UdeA — Per-Unit Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '10_per_unit_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 10_per_unit_performance.png")

    unit_df_result.to_csv(os.path.join(VIZ_DIR, 'per_unit_metrics.csv'), index=False)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================
def save_classification_report(all_preds):
    """Save sklearn classification report for best model."""
    best = '1_LogisticRegression_L2'
    if best not in all_preds:
        return
    df = all_preds[best]
    report = classification_report(df['y_true'], df['y_pred'],
                                    target_names=['Sano (0)', 'Riesgo (1)'],
                                    digits=4)
    report_path = os.path.join(VIZ_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("FIRE-UdeA v2.0 — Classification Report (LogReg L2)\n")
        f.write("=" * 60 + "\n")
        f.write(f"All LOYO out-of-fold predictions (n={len(df)})\n\n")
        f.write(report)
    print(f"  ✓ classification_report.txt")
    print(f"\n{report}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("FIRE-UdeA v2.0 — COMPREHENSIVE METRICS & VISUALIZATIONS")
    print("=" * 70)

    # Load data
    all_preds = load_predictions()
    metrics_df = load_metrics()
    print(f"\nLoaded {len(all_preds)} models, {len(metrics_df)} metric rows")

    # Generate everything
    metrics_table = compute_full_metrics(all_preds)
    plot_confusion_matrices(all_preds)
    plot_roc_curves(all_preds, metrics_df)
    plot_pr_curves(all_preds)
    plot_per_fold_auc(metrics_df, all_preds)
    plot_probability_distributions(all_preds)
    plot_prediction_heatmaps(all_preds)
    plot_radar_chart(metrics_table)
    plot_threshold_analysis(all_preds)
    plot_train_test_gap(metrics_df, all_preds)
    plot_per_unit_performance(all_preds)
    save_classification_report(all_preds)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — All outputs in: {VIZ_DIR}")
    print(f"{'=' * 70}")
    for f in sorted(os.listdir(VIZ_DIR)):
        size = os.path.getsize(os.path.join(VIZ_DIR, f))
        print(f"  {f} ({size:,} bytes)")


if __name__ == '__main__':
    main()
