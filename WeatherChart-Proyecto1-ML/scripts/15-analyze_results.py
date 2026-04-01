"""
15-analyze_results.py
Comprehensive post-training analysis of the XGBoost Music Genre classifier.

This script loads the trained model and test data, then generates
professional-grade visualizations for academic presentation:

  1. Per-class F1 Score ranking (horizontal bar)
  2. Confusion heatmap — most confused genre pairs
  3. Prediction confidence distribution (probability histograms)
  4. Genre accuracy by continent (geographic patterns)
  5. Error analysis — detailed breakdown of model mistakes
  6. Precision vs Recall scatter (per class)
  7. Cumulative accuracy curve (how many classes above X% accuracy)

Outputs:
  - plots/f1_per_class.png
  - plots/top_confusions.png
  - plots/confidence_distribution.png
  - plots/accuracy_by_continent.png
  - plots/error_analysis.png
  - plots/precision_recall_scatter.png
  - plots/cumulative_accuracy.png
  - data/analysis_report.txt
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Visual style
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

PALETTE = sns.color_palette("viridis", 20)


# ============================================================
# MAIN
# ============================================================
def main():
    report = []

    def log(msg):
        print(msg)
        report.append(msg)

    log("=" * 60)
    log("15-ANALYZE_RESULTS.PY — Post-Training Analysis")
    log("=" * 60)

    # ----------------------------------------------------------
    # LOAD DATA & MODEL
    # ----------------------------------------------------------
    log("\n[LOAD] Loading model, data, and artifacts...")

    model = XGBClassifier()
    model.load_model(os.path.join(DATA_DIR, 'xgboost_model.json'))

    X_test = pd.read_parquet(os.path.join(DATA_DIR, 'X_test.parquet'))
    y_test = pd.read_parquet(os.path.join(DATA_DIR, 'y_test.parquet')).iloc[:, 0]

    with open(os.path.join(DATA_DIR, 'preprocessing_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)

    target_classes = np.array(artifacts['target_classes'])
    feature_names = artifacts['feature_names']
    encoders = artifacts['encoders']

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    log(f"  Test samples: {len(y_test):,}")
    log(f"  Classes:      {len(target_classes)}")
    log(f"  Features:     {len(feature_names)}")

    # ----------------------------------------------------------
    # PLOT 1: Per-Class F1 Score Ranking
    # ----------------------------------------------------------
    log("\n[PLOT 1] Per-class F1 score ranking...")

    report_dict = classification_report(
        y_test, y_pred,
        target_names=target_classes,
        output_dict=True,
        zero_division=0,
    )

    class_metrics = []
    for cls in target_classes:
        if cls in report_dict:
            m = report_dict[cls]
            class_metrics.append({
                'genre': cls,
                'f1': m['f1-score'],
                'precision': m['precision'],
                'recall': m['recall'],
                'support': m['support'],
            })

    df_metrics = pd.DataFrame(class_metrics).sort_values('f1', ascending=True)

    # Show bottom 30 (most interesting — where model struggles)
    df_bottom = df_metrics.head(30)

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ['#e74c3c' if f < 0.95 else '#f39c12' if f < 0.99 else '#2ecc71'
              for f in df_bottom['f1']]
    ax.barh(df_bottom['genre'], df_bottom['f1'], color=colors)
    ax.set_xlim(0.90, 1.005)
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score by Genre (Bottom 30 — Zoomed In)', fontweight='bold')
    ax.axvline(x=0.99, color='gray', linestyle='--', alpha=0.5, label='0.99 threshold')
    ax.legend()

    # Annotate values
    for i, (_, row) in enumerate(df_bottom.iterrows()):
        ax.text(row['f1'] + 0.0005, i, f"{row['f1']:.4f}", va='center', fontsize=8)

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'f1_per_class.png'), dpi=150)
    plt.close()
    log("  Saved: f1_per_class.png")

    # ----------------------------------------------------------
    # PLOT 2: Top Confusion Pairs
    # ----------------------------------------------------------
    log("\n[PLOT 2] Most confused genre pairs...")

    cm = confusion_matrix(y_test, y_pred)
    np.fill_diagonal(cm, 0)  # Zero out correct predictions

    # Find top 15 confusion pairs
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j] > 0:
                confusions.append({
                    'true': target_classes[i],
                    'predicted': target_classes[j],
                    'count': cm[i, j],
                })

    df_conf = pd.DataFrame(confusions).sort_values('count', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [f"{row['true']} → {row['predicted']}" for _, row in df_conf.iterrows()]
    colors = plt.cm.OrRd(np.linspace(0.3, 0.9, len(df_conf)))[::-1]
    ax.barh(labels[::-1], df_conf['count'].values[::-1], color=colors)
    ax.set_xlabel('Number of Misclassifications')
    ax.set_title('Top 15 Most Confused Genre Pairs', fontweight='bold')

    for i, v in enumerate(df_conf['count'].values[::-1]):
        ax.text(v + 0.3, i, str(int(v)), va='center', fontsize=10, fontweight='bold')

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_confusions.png'), dpi=150)
    plt.close()
    log("  Saved: top_confusions.png")

    # ----------------------------------------------------------
    # PLOT 3: Prediction Confidence Distribution
    # ----------------------------------------------------------
    log("\n[PLOT 3] Prediction confidence distribution...")

    max_proba = y_proba.max(axis=1)
    correct = y_pred == y_test.values
    wrong = ~correct

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    axes[0].hist(max_proba, bins=50, color='#3498db', alpha=0.8, edgecolor='white')
    axes[0].set_xlabel('Max Predicted Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Overall Confidence Distribution', fontweight='bold')
    axes[0].axvline(x=np.median(max_proba), color='red', linestyle='--',
                    label=f'Median: {np.median(max_proba):.4f}')
    axes[0].legend()

    # Correct vs Incorrect
    if wrong.sum() > 0:
        axes[1].hist(max_proba[correct], bins=50, alpha=0.7, label='Correct',
                     color='#2ecc71', edgecolor='white')
        axes[1].hist(max_proba[wrong], bins=50, alpha=0.7, label='Incorrect',
                     color='#e74c3c', edgecolor='white')
        axes[1].set_xlabel('Max Predicted Probability')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Confidence: Correct vs Incorrect', fontweight='bold')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No incorrect predictions!',
                     ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('Confidence: Correct vs Incorrect', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confidence_distribution.png'), dpi=150)
    plt.close()
    log("  Saved: confidence_distribution.png")

    log(f"  Confidence stats:")
    log(f"    Mean:   {max_proba.mean():.6f}")
    log(f"    Median: {np.median(max_proba):.6f}")
    log(f"    Min:    {max_proba.min():.6f}")
    log(f"    <0.9:   {(max_proba < 0.9).sum()} samples")
    log(f"    <0.5:   {(max_proba < 0.5).sum()} samples")

    # ----------------------------------------------------------
    # PLOT 4: Accuracy by Continent
    # ----------------------------------------------------------
    log("\n[PLOT 4] Accuracy by continent...")

    le_continent = encoders.get('continent')
    if le_continent is not None:
        continent_encoded = X_test['continent'].values
        continent_names = le_continent.inverse_transform(
            continent_encoded.astype(int)
        )

        df_geo = pd.DataFrame({
            'continent': continent_names,
            'correct': correct,
        })

        continent_stats = df_geo.groupby('continent').agg(
            accuracy=('correct', 'mean'),
            total=('correct', 'count'),
            errors=('correct', lambda x: (~x).sum()),
        ).sort_values('accuracy', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if a < 0.99 else '#2ecc71' for a in continent_stats['accuracy']]
        bars = ax.barh(continent_stats.index, continent_stats['accuracy'], color=colors)
        ax.set_xlim(0.995, 1.001)
        ax.set_xlabel('Accuracy')
        ax.set_title('Model Accuracy by Continent', fontweight='bold')

        for i, (idx, row) in enumerate(continent_stats.iterrows()):
            ax.text(row['accuracy'] + 0.0001, i,
                    f"{row['accuracy']:.4f} ({int(row['errors'])} errors / {int(row['total']):,})",
                    va='center', fontsize=9)

        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_by_continent.png'), dpi=150)
        plt.close()
        log("  Saved: accuracy_by_continent.png")

        log(f"\n  Continent breakdown:")
        for idx, row in continent_stats.iterrows():
            log(f"    {idx:<25} {row['accuracy']:.4f}  "
                f"({int(row['errors'])} errors / {int(row['total']):,} samples)")

    # ----------------------------------------------------------
    # PLOT 5: Error Analysis
    # ----------------------------------------------------------
    log("\n[PLOT 5] Error analysis...")

    n_errors = int(wrong.sum())
    n_total = len(y_test)
    log(f"  Total errors: {n_errors} / {n_total:,} ({n_errors/n_total*100:.4f}%)")

    if n_errors > 0:
        error_idx = np.where(wrong)[0]
        true_labels = target_classes[y_test.values[error_idx]]
        pred_labels = target_classes[y_pred[error_idx]]
        error_confidence = max_proba[error_idx]

        # Which genres are hardest to classify?
        error_by_genre = pd.Series(true_labels).value_counts().head(15)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Errors by true genre
        axes[0].barh(error_by_genre.index[::-1], error_by_genre.values[::-1],
                     color='#e74c3c', alpha=0.8)
        axes[0].set_xlabel('Number of Errors')
        axes[0].set_title('Genres with Most Errors (True Label)', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)

        for i, v in enumerate(error_by_genre.values[::-1]):
            axes[0].text(v + 0.1, i, str(v), va='center', fontsize=10)

        # Confidence of wrong predictions
        axes[1].hist(error_confidence, bins=30, color='#e74c3c', alpha=0.8,
                     edgecolor='white')
        axes[1].set_xlabel('Confidence of Wrong Prediction')
        axes[1].set_ylabel('Count')
        axes[1].set_title('How Confident Were the Mistakes?', fontweight='bold')
        axes[1].axvline(x=np.median(error_confidence), color='black',
                        linestyle='--', label=f'Median: {np.median(error_confidence):.4f}')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'error_analysis.png'), dpi=150)
        plt.close()
        log("  Saved: error_analysis.png")

        # Print detailed error table
        log("\n  Detailed error breakdown (all misclassifications):")
        log(f"  {'True Genre':<25} {'Predicted As':<25} {'Confidence':>10}")
        log(f"  {'-'*60}")
        for i in range(min(n_errors, 30)):
            log(f"  {true_labels[i]:<25} {pred_labels[i]:<25} "
                f"{error_confidence[i]:>10.6f}")
        if n_errors > 30:
            log(f"  ... and {n_errors - 30} more errors.")
    else:
        log("  No errors found — perfect classification!")

    # ----------------------------------------------------------
    # PLOT 6: Precision vs Recall Scatter
    # ----------------------------------------------------------
    log("\n[PLOT 6] Precision vs Recall scatter...")

    fig, ax = plt.subplots(figsize=(10, 8))

    sizes = df_metrics['support'] / df_metrics['support'].max() * 300 + 20

    scatter = ax.scatter(
        df_metrics['precision'], df_metrics['recall'],
        s=sizes.values, c=df_metrics['f1'], cmap='RdYlGn',
        vmin=0.95, vmax=1.0, alpha=0.8, edgecolors='gray', linewidths=0.5,
    )

    # Label genres with F1 < 0.998
    for _, row in df_metrics[df_metrics['f1'] < 0.998].iterrows():
        ax.annotate(row['genre'], (row['precision'], row['recall']),
                    fontsize=8, ha='center', va='bottom',
                    textcoords='offset points', xytext=(0, 6))

    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Precision vs Recall per Genre\n(size = sample count, color = F1)',
                 fontweight='bold')
    ax.set_xlim(0.96, 1.005)
    ax.set_ylim(0.96, 1.005)
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_scatter.png'), dpi=150)
    plt.close()
    log("  Saved: precision_recall_scatter.png")

    # ----------------------------------------------------------
    # PLOT 7: Cumulative Accuracy Curve
    # ----------------------------------------------------------
    log("\n[PLOT 7] Cumulative accuracy curve...")

    per_class_accuracy = []
    for i, cls in enumerate(target_classes):
        mask = y_test.values == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            per_class_accuracy.append(acc)

    sorted_acc = np.sort(per_class_accuracy)
    cumulative = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(sorted_acc, cumulative, where='post', linewidth=2.5, color='#2196F3')
    ax.fill_between(sorted_acc, cumulative, step='post', alpha=0.15, color='#2196F3')
    ax.set_xlabel('Per-Class Accuracy')
    ax.set_ylabel('Cumulative Proportion of Classes')
    ax.set_title('Cumulative Accuracy Curve\n(What % of genres achieve X% accuracy?)',
                 fontweight='bold')
    ax.set_xlim(0.95, 1.005)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0.99, color='red', linestyle='--', alpha=0.5,
               label=f'99% threshold')

    # Annotate
    above_99 = (np.array(per_class_accuracy) >= 0.99).sum()
    ax.text(0.99, 0.5, f'{above_99}/{len(per_class_accuracy)} genres\nabove 99%',
            fontsize=11, color='red', ha='left', va='center',
            transform=ax.get_xaxis_transform())
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cumulative_accuracy.png'), dpi=150)
    plt.close()
    log("  Saved: cumulative_accuracy.png")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("ANALYSIS COMPLETE — 7 plots generated")
    log("=" * 60)
    log(f"  1. f1_per_class.png          — Per-genre F1 ranking")
    log(f"  2. top_confusions.png        — Most confused genre pairs")
    log(f"  3. confidence_distribution.png — Prediction confidence")
    log(f"  4. accuracy_by_continent.png — Geographic accuracy")
    log(f"  5. error_analysis.png        — Error breakdown")
    log(f"  6. precision_recall_scatter.png — P/R per class")
    log(f"  7. cumulative_accuracy.png   — Cumulative accuracy curve")
    log("=" * 60)

    # Save report
    report_path = os.path.join(DATA_DIR, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    log(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
