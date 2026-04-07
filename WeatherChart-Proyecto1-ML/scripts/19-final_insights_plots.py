"""
19-final_insights_plots.py
Advanced Data Insights & Migration Visualizations.

Purpose:
  Visualize the "Noise Flow" and "Genre Purification" resulting from 
  the unsupervised label re-evaluation process.

Plots:
  1. Label Migration Matrix (Heatmap)
  2. Genre Fidelity (Most Changed vs. Most Stable)
  3. Genre Radar Charts (ADN Comparison: Before vs. After)

Outputs:
  - Plots generated in General/Presentacion_WeatherChart/plots/
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.metrics import confusion_matrix

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

# Features to use in the Radar Charts (must exist in X_train)
RADAR_FEATURES = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'valence', 'tempo', 'climatology', 'gdp_per_capita'
]

# Genres to compare in Radar Charts (must exist in classes list)
RADAR_GENRES_QUERY = ['pop', 'rock', 'latin', 'reggae', 'electro']

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FAFAFA',
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
})

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def plot_radar(genre_name, before_vals, after_vals, labels, filepath):
    """Generate a side-by-side radar (spider) plot comparison."""
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop
    before_vals = np.concatenate((before_vals, [before_vals[0]]))
    after_vals = np.concatenate((after_vals, [after_vals[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot Before (Original Noisy)
    ax.plot(angles, before_vals, color='#e74c3c', linewidth=2, linestyle='--', label='Original (Noisy)')
    ax.fill(angles, before_vals, color='#e74c3c', alpha=0.15)
    
    # Plot After (Cleaned)
    ax.plot(angles, after_vals, color='#2ecc71', linewidth=3, label='Cleaned (Relabeled)')
    ax.fill(angles, after_vals, color='#2ecc71', alpha=0.3)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    
    ax.set_title(f'Genre Purification: {genre_name.upper()}', weight='bold', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 65)
    print("19-FINAL_INSIGHTS_PLOTS.PY — Advanced Analytics")
    print("=" * 65)

    # 1. Load Data
    print("\n[STEP 1] Loading Labels and Artifacts...")
    y_orig_df = pd.read_parquet(os.path.join(DATA_DIR, 'y_train.parquet'))
    y_relabel_df = pd.read_parquet(os.path.join(DATA_DIR, 'y_train_relabeled.parquet'))
    
    y_orig = y_orig_df.iloc[:, 0].values
    y_new = y_relabel_df.iloc[:, 0].values

    with open(os.path.join(DATA_DIR, 'preprocessing_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)
    classes = artifacts['target_classes']
    
    # 2. Label Migration Matrix (Migration Flow)
    print("\n[STEP 2] Generating Label Migration Heatmap...")
    counts = pd.Series(y_orig).value_counts()
    top_20_indices = counts.head(20).index.tolist()
    top_20_names = [classes[i] for i in top_20_indices]
    
    mask = np.isin(y_orig, top_20_indices) & np.isin(y_new, top_20_indices)
    cm = confusion_matrix(y_orig[mask], y_new[mask], labels=top_20_indices)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=False, cmap='YlGnBu', xticklabels=top_20_names, yticklabels=top_20_names)
    plt.title('Label Migration Matrix (Top 20 Genres)\nNoise Flow from Original to Cleaned Labels', fontweight='bold')
    plt.xlabel('Cleaned (Relabeled) Genre')
    plt.ylabel('Original (Noisy) Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'insight_migration_heatmap.png'), dpi=150)
    plt.close()
    print("  Saved: insight_migration_heatmap.png")

    # 3. Genre Fidelity Ranking
    print("\n[STEP 3] Generating Genre Fidelity Ranking...")
    fidelity = []
    for i, name in enumerate(classes):
        mask = y_orig == i
        if mask.sum() > 0:
            changed = (y_orig[mask] != y_new[mask]).mean()
            fidelity.append({'Genre': name, 'Change_Rate': changed, 'Count': mask.sum()})
    
    fid_df = pd.DataFrame(fidelity).sort_values('Change_Rate', ascending=False)
    
    # Top 15 most unstable
    plt.figure(figsize=(10, 8))
    sns.barplot(data=fid_df.head(15), x='Change_Rate', y='Genre', palette='Reds_r')
    plt.title('Top 15 Most Unstable (Noisiest) Genres\nPercentage of labels changed by Clustering', fontweight='bold')
    plt.xlabel('Re-label Rate (1.0 = 100% changed)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'insight_unstable_genres.png'), dpi=150)
    plt.close()
    
    # Top 15 most stable
    plt.figure(figsize=(10, 8))
    sns.barplot(data=fid_df.tail(15), x='Change_Rate', y='Genre', palette='Greens_r')
    plt.title('Top 15 Most Stable (Standardized) Genres\nWhich genres matched their mathematical density?', fontweight='bold')
    plt.xlabel('Re-label Rate (Lower is more stable)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'insight_stable_genres.png'), dpi=150)
    plt.close()
    print("  Saved: unstable/stable ranking plots.")

    # 4. Radar Charts (Genre ADN)
    print("\n[STEP 4] Generating Genre Purification Radar Charts (ADN Comparison)...")
    X_train = pd.read_parquet(os.path.join(DATA_DIR, 'X_train.parquet'))
    X_cols = X_train.columns
    
    # Filter features that exist
    features = [f for f in RADAR_FEATURES if f in X_cols]
    X_radar = X_train[features]
    
    # We use min-max for visual clarity in radar plots (relative profile)
    X_norm = (X_radar - X_radar.min()) / (X_radar.max() - X_radar.min() + 1e-10)
    
    for query in RADAR_GENRES_QUERY:
        # Find exact class index
        found_idx = -1
        for i, name in enumerate(classes):
            if query.lower() in name.lower():
                found_idx = i
                break
        
        if found_idx == -1: continue
        
        mask_orig = y_orig == found_idx
        mask_new = y_new == found_idx
        
        if mask_orig.sum() < 50 or mask_new.sum() < 50:
            continue
            
        before_means = X_norm[mask_orig].mean().values
        after_means = X_norm[mask_new].mean().values
        
        fname = f'insight_radar_{classes[found_idx].replace("-","_")}.png'
        plot_radar(classes[found_idx], before_means, after_means, features, os.path.join(PLOTS_DIR, fname))
        print(f"  Saved: {fname}")

    print("\n" + "=" * 65)
    print("INSIGHTS PIPELINE COMPLETE")
    print("=" * 65)

if __name__ == '__main__':
    main()
