"""
16-unsupervised_clustering.py
Unsupervised Clustering Analysis & Label Re-evaluation Pipeline.

Purpose:
  Apply 5 clustering algorithms to discover natural data groupings and
  re-evaluate the original genre labels, following the principle that
  ~30% of labels may be incorrectly assigned.

Algorithms:
  1. K-Means
  2. Fuzzy C-Means  (custom implementation)
  3. Subtractive Clustering (custom implementation)
  4. DBSCAN
  5. Agglomerative Clustering

Outputs:
  - data/y_train_relabeled.parquet     (re-labeled target vector)
  - data/clustering_report.txt         
  - Plots generated in General/Presentacion_WeatherChart/plots/
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
import umap.umap_ as umap

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'General', 'Presentacion_WeatherChart', 'plots')

SAMPLE_CLUSTERING  = 30_000   
SAMPLE_SUBTRACTIVE = 5_000    
SAMPLE_SILHOUETTE  = 8_000    

FCM_FUZZINESS = 2.0
FCM_MAX_ITER  = 200
FCM_TOL       = 1e-6

SUB_RA           = 0.5     
SUB_RB_FACTOR    = 1.5     
SUB_ACCEPT_RATIO = 0.5     
SUB_REJECT_RATIO = 0.15    

DBSCAN_MIN_SAMPLES = 10

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FAFAFA',
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
})

RANDOM_STATE = 42

# ============================================================
# CUSTOM IMPLEMENTATIONS
# ============================================================
def fuzzy_cmeans(X, n_clusters, m=2.0, max_iter=200, tol=1e-6, random_state=42):
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    power = 2.0 / (m - 1.0)

    U = rng.rand(n, n_clusters).astype(np.float64)
    U /= U.sum(axis=1, keepdims=True)

    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        Um = U ** m
        sum_Um = np.maximum(Um.sum(axis=0), 1e-16)
        centers = (Um.T @ X) / sum_Um[:, np.newaxis]
        distances = np.maximum(cdist(X, centers, metric='euclidean'), 1e-10)
        d_neg_pow = distances ** (-power)
        U_new = d_neg_pow / d_neg_pow.sum(axis=1, keepdims=True)
        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new
    return np.argmax(U, axis=1), centers, U, n_iter

def subtractive_clustering(X, ra=0.5, rb_factor=1.5, accept_ratio=0.5, reject_ratio=0.15):
    n, d = X.shape
    rb = ra * rb_factor
    alpha = 4.0 / (ra ** 2)
    beta  = 4.0 / (rb ** 2)

    potentials = np.zeros(n)
    batch_size = min(1000, n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dist_sq = cdist(X[start:end], X, metric='sqeuclidean')
        potentials[start:end] = np.exp(-alpha * dist_sq).sum(axis=1)

    centers_idx = []
    first_potential = potentials.max()
    max_centers = min(200, n // 5)

    while True:
        best_idx = np.argmax(potentials)
        best_potential = potentials[best_idx]
        if first_potential < 1e-10: break
        ratio = best_potential / first_potential
        if ratio < reject_ratio: break
        
        accept = False
        if ratio > accept_ratio:
            accept = True
        else:
            if centers_idx:
                d_min = np.sqrt(np.sum((X[centers_idx] - X[best_idx]) ** 2, axis=1)).min()
                if d_min / ra + ratio >= 1: accept = True
                else:
                    potentials[best_idx] = 0
                    continue
            else: accept = True

        if accept:
            centers_idx.append(best_idx)
            dist_sq = np.sum((X - X[best_idx]) ** 2, axis=1)
            potentials -= best_potential * np.exp(-beta * dist_sq)
            potentials = np.maximum(potentials, 0)

        if len(centers_idx) >= max_centers: break

    if len(centers_idx) == 0:
        return np.zeros(n, dtype=int), X.mean(axis=0).reshape(1, -1), 1
    center_coords = X[centers_idx]
    dist_to_centers = cdist(X, center_coords, metric='euclidean')
    labels = np.argmin(dist_to_centers, axis=1)
    return labels, center_coords, len(centers_idx)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def cluster_to_label_map(cluster_labels, true_labels):
    mapping = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if mask.sum() > 0:
            vals, counts = np.unique(true_labels[mask], return_counts=True)
            mapping[c] = vals[np.argmax(counts)]
    return mapping

def relabel_by_majority(cluster_labels, true_labels):
    mapping = cluster_to_label_map(cluster_labels, true_labels)
    new_labels = np.array([mapping.get(c, l) for c, l in zip(cluster_labels, true_labels)])
    return new_labels, int((new_labels != true_labels).sum())

def safe_silhouette(X, labels, sample_size=8000, random_state=42):
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(X): return float('nan')
    if len(X) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), sample_size, replace=False)
        return silhouette_score(X[idx], labels[idx])
    return silhouette_score(X, labels)

def safe_davies_bouldin(X, labels):
    if len(np.unique(labels)) < 2: return float('nan')
    return davies_bouldin_score(X, labels)

def plot_2d_clusters(data_2d, labels, title, filepath, xlabel='Dim 1', ylabel='Dim 2', n_clusters_show=15):
    fig, ax = plt.subplots(figsize=(10, 8))
    counts = pd.Series(labels).value_counts()
    top_clusters = counts.head(n_clusters_show).index.tolist()
    cmap = cm.get_cmap('tab20', min(n_clusters_show, 20))
    
    other_mask = ~np.isin(labels, top_clusters)
    if other_mask.sum() > 0:
        ax.scatter(data_2d[other_mask, 0], data_2d[other_mask, 1], c='#CCCCCC', s=3, alpha=0.2, label='Other')
        
    for i, c in enumerate(top_clusters):
        mask = labels == c
        ax.scatter(data_2d[mask, 0], data_2d[mask, 1], c=[cmap(i % 20)], s=5, alpha=0.5, label=f'C {c} ({mask.sum():,})')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2, markerscale=3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def plot_3d_clusters(data_3d, labels, title, filepath, n_clusters_show=15):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    counts = pd.Series(labels).value_counts()
    top_clusters = counts.head(n_clusters_show).index.tolist()
    cmap = cm.get_cmap('tab20', min(n_clusters_show, 20))
    
    other_mask = ~np.isin(labels, top_clusters)
    if other_mask.sum() > 0:
        ax.scatter(data_3d[other_mask, 0], data_3d[other_mask, 1], data_3d[other_mask, 2],
                   c='#CCCCCC', s=3, alpha=0.1, label='Other')
                   
    for i, c in enumerate(top_clusters):
        mask = labels == c
        ax.scatter(data_3d[mask, 0], data_3d[mask, 1], data_3d[mask, 2],
                   c=[cmap(i % 20)], s=5, alpha=0.5, label=f'C {c} ({mask.sum():,})')
                   
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2, markerscale=3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    report = []
    def log(msg):
        print(msg)
        report.append(msg)

    log("=" * 65)
    log("16-UNSUPERVISED_CLUSTERING.PY — Clustering & Label Re-evaluation")
    log("=" * 65)

    log("\n[STEP 1] Loading preprocessed training data...")
    X_train_full = pd.read_parquet(os.path.join(DATA_DIR, 'X_train.parquet'))
    y_train_full = pd.read_parquet(os.path.join(DATA_DIR, 'y_train.parquet')).iloc[:, 0].values

    with open(os.path.join(DATA_DIR, 'preprocessing_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)

    target_classes = np.array(artifacts['target_classes'])
    n_classes = len(target_classes)

    nan_counts = X_train_full.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        X_train_full = pd.DataFrame(imputer.fit_transform(X_train_full), columns=X_train_full.columns)

    X_full = X_train_full.values

    log("\n[STEP 2] Creating stratified sample for clustering evaluation...")
    rng = np.random.RandomState(RANDOM_STATE)
    n_total = len(X_full)

    n_sample = min(SAMPLE_CLUSTERING, n_total)
    idx_main = rng.choice(n_total, n_sample, replace=False)
    X_sample, y_sample = X_full[idx_main], y_train_full[idx_main]

    n_sub = min(SAMPLE_SUBTRACTIVE, n_total)
    idx_sub = rng.choice(n_total, n_sub, replace=False)
    X_sub, y_sub = X_full[idx_sub], y_train_full[idx_sub]

    log("\n[STEP 3] Computing PCA and UMAP projections for visualisation...")
    
    # PCA
    pca_3d_mod = PCA(n_components=3, random_state=RANDOM_STATE)
    pca_3d = pca_3d_mod.fit_transform(X_sample)
    pca_2d = pca_3d[:, :2]
    pca_3d_sub = pca_3d_mod.transform(X_sub)
    pca_2d_sub = pca_3d_sub[:, :2]
    
    # UMAP
    log("  Computing UMAP embeddings... This may take a moment.")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    umap_2d = reducer.fit_transform(X_sample)
    umap_2d_sub = reducer.transform(X_sub)

    plot_2d_clusters(pca_2d, y_sample, 'PCA 2D — Original Labels', os.path.join(PLOTS_DIR, 'clustering_pca_original.png'), 'PC1', 'PC2')
    plot_3d_clusters(pca_3d, y_sample, 'PCA 3D — Original Labels', os.path.join(PLOTS_DIR, 'clustering_pca3d_original.png'))
    plot_2d_clusters(umap_2d, y_sample, 'UMAP 2D — Original Labels', os.path.join(PLOTS_DIR, 'clustering_umap_original.png'), 'UMAP1', 'UMAP2')

    results = {}

    log(f"\n[STEP 4a] K-Means (k={n_classes})...")
    t0 = time.time()
    km = KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init=3, max_iter=300)
    km_labels = km.fit_predict(X_sample)
    results['K-Means'] = {'labels': km_labels, 'time': time.time() - t0, 'X': X_sample, 'y': y_sample}
    plot_2d_clusters(pca_2d, km_labels, f'K-Means (k={n_classes}) [PCA]', os.path.join(PLOTS_DIR, 'clustering_kmeans_pca2d.png'), 'PC1', 'PC2')
    plot_3d_clusters(pca_3d, km_labels, f'K-Means (k={n_classes}) [PCA 3D]', os.path.join(PLOTS_DIR, 'clustering_kmeans_pca3d.png'))
    plot_2d_clusters(umap_2d, km_labels, f'K-Means (k={n_classes}) [UMAP]', os.path.join(PLOTS_DIR, 'clustering_kmeans_umap.png'), 'UMAP1', 'UMAP2')

    log(f"\n[STEP 4b] Fuzzy C-Means (k={n_classes}, m={FCM_FUZZINESS})...")
    t0 = time.time()
    fcm_labels, _, _, _ = fuzzy_cmeans(X_sample, n_classes, m=FCM_FUZZINESS, max_iter=FCM_MAX_ITER, tol=FCM_TOL, random_state=RANDOM_STATE)
    results['Fuzzy C-Means'] = {'labels': fcm_labels, 'time': time.time() - t0, 'X': X_sample, 'y': y_sample}
    plot_2d_clusters(pca_2d, fcm_labels, f'FCM [PCA]', os.path.join(PLOTS_DIR, 'clustering_fcm_pca2d.png'), 'PC1', 'PC2')
    plot_3d_clusters(pca_3d, fcm_labels, f'FCM [PCA 3D]', os.path.join(PLOTS_DIR, 'clustering_fcm_pca3d.png'))
    plot_2d_clusters(umap_2d, fcm_labels, f'FCM [UMAP]', os.path.join(PLOTS_DIR, 'clustering_fcm_umap.png'), 'UMAP1', 'UMAP2')

    log(f"\n[STEP 4c] Subtractive Clustering (ra={SUB_RA})...")
    t0 = time.time()
    sub_labels, _, sub_k = subtractive_clustering(X_sub, ra=SUB_RA, rb_factor=SUB_RB_FACTOR, accept_ratio=SUB_ACCEPT_RATIO, reject_ratio=SUB_REJECT_RATIO)
    results['Subtractive'] = {'labels': sub_labels, 'time': time.time() - t0, 'X': X_sub, 'y': y_sub}
    plot_2d_clusters(pca_2d_sub, sub_labels, f'Subtractive (k={sub_k}) [PCA]', os.path.join(PLOTS_DIR, 'clustering_subtractive_pca2d.png'), 'PC1', 'PC2')
    plot_3d_clusters(pca_3d_sub, sub_labels, f'Subtractive (k={sub_k}) [PCA 3D]', os.path.join(PLOTS_DIR, 'clustering_subtractive_pca3d.png'))
    plot_2d_clusters(umap_2d_sub, sub_labels, f'Subtractive (k={sub_k}) [UMAP]', os.path.join(PLOTS_DIR, 'clustering_subtractive_umap.png'), 'UMAP1', 'UMAP2')

    log(f"\n[STEP 4d] DBSCAN (auto-eps, min_samples={DBSCAN_MIN_SAMPLES})...")
    nn = NearestNeighbors(n_neighbors=DBSCAN_MIN_SAMPLES)
    distances_knn, _ = nn.fit(X_sample).kneighbors(X_sample)
    eps_estimated = np.percentile(np.sort(distances_knn[:, -1]), 90)
    t0 = time.time()
    db_labels = DBSCAN(eps=eps_estimated, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit_predict(X_sample)
    results['DBSCAN'] = {'labels': db_labels, 'time': time.time() - t0, 'X': X_sample, 'y': y_sample}
    plot_2d_clusters(pca_2d, db_labels, f'DBSCAN [PCA]', os.path.join(PLOTS_DIR, 'clustering_dbscan_pca2d.png'), 'PC1', 'PC2')
    plot_3d_clusters(pca_3d, db_labels, f'DBSCAN [PCA 3D]', os.path.join(PLOTS_DIR, 'clustering_dbscan_pca3d.png'))
    plot_2d_clusters(umap_2d, db_labels, f'DBSCAN [UMAP]', os.path.join(PLOTS_DIR, 'clustering_dbscan_umap.png'), 'UMAP1', 'UMAP2')

    log(f"\n[STEP 4e] Agglomerative Clustering (k={n_classes})...")
    connectivity = kneighbors_graph(X_sample, n_neighbors=10, include_self=False)
    t0 = time.time()
    agg_labels = AgglomerativeClustering(n_clusters=n_classes, connectivity=connectivity, linkage='ward').fit_predict(X_sample)
    results['Agglomerative'] = {'labels': agg_labels, 'time': time.time() - t0, 'X': X_sample, 'y': y_sample}
    plot_2d_clusters(pca_2d, agg_labels, f'Agglomerative [PCA]', os.path.join(PLOTS_DIR, 'clustering_agglomerative_pca2d.png'), 'PC1', 'PC2')
    plot_3d_clusters(pca_3d, agg_labels, f'Agglomerative [PCA 3D]', os.path.join(PLOTS_DIR, 'clustering_agglomerative_pca3d.png'))
    plot_2d_clusters(umap_2d, agg_labels, f'Agglomerative [UMAP]', os.path.join(PLOTS_DIR, 'clustering_agglomerative_umap.png'), 'UMAP1', 'UMAP2')

    log("\n[STEP 5] Computing metrics...")
    metrics_table = []
    for name, res in results.items():
        vld = res['labels'] >= 0
        sil = safe_silhouette(res['X'], res['labels'], SAMPLE_SILHOUETTE)
        dbi = safe_davies_bouldin(res['X'], res['labels'])
        ari = adjusted_rand_score(res['y'][vld], res['labels'][vld]) if vld.sum() > 0 else float('nan')
        nmi = normalized_mutual_info_score(res['y'][vld], res['labels'][vld]) if vld.sum() > 0 else float('nan')
        metrics_table.append({'Algorithm': name, 'Silhouette': sil, 'D-B': dbi, 'ARI': ari, 'NMI': nmi})

    log("\n[STEP 6] Re-labeling using MiniBatch K-Means...")
    mbkm = MiniBatchKMeans(n_clusters=n_classes, random_state=RANDOM_STATE, batch_size=2048, n_init=3, max_iter=300)
    full_cluster_labels = mbkm.fit_predict(X_full)
    new_labels, n_changed = relabel_by_majority(full_cluster_labels, y_train_full)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].pie([len(y_train_full) - n_changed, n_changed], labels=['Unchanged', 'Re-labeled'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Label Re-evaluation Summary', fontweight='bold')
    plt.savefig(os.path.join(PLOTS_DIR, 'relabeling_summary.png'), dpi=150)
    plt.close()

    y_relabeled_df = pd.DataFrame({'primary_genre': new_labels})
    y_relabeled_df.to_parquet(os.path.join(DATA_DIR, 'y_train_relabeled.parquet'), index=False)
    
    with open(os.path.join(DATA_DIR, 'clustering_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
if __name__ == '__main__':
    main()
