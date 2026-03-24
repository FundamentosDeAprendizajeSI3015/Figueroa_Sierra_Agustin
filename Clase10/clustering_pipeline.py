import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid", context="talk")

# ---------------------------------------------------------
# SUBTRACTIVE CLUSTERING (Chiu 1994)
# ---------------------------------------------------------
def subtractive_clustering(X, ra=0.4, rb=0.6, eps_upper=0.5, eps_lower=0.15, max_iter=1000):
    """
    Subtractive clustering algorithm from scratch.
    """
    n_samples, n_features = X.shape
    
    # Precompute pairwise squared distances
    distances_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    
    # Calculate density potentials
    potentials = np.sum(np.exp(-4 * distances_sq / (ra ** 2)), axis=1)
    
    centers = []
    center_indices = []
    P1 = np.max(potentials)
    
    for _ in range(max_iter):
        idx_max = np.argmax(potentials)
        P_max = potentials[idx_max]
        
        if P_max > eps_upper * P1:
            # Accept center
            pass
        elif P_max < eps_lower * P1:
            # Reject all future and stop
            break
        else:
            # Check acceptance condition (distance vs potential threshold)
            if not centers:
                break
            d_min = np.min(np.sqrt(np.sum((centers - X[idx_max])**2, axis=1)))
            if (d_min / ra) + (P_max / P1) >= 1.0:
                # Accept
                pass
            else:
                # Reject this point by setting its potential to 0 and find next best
                potentials[idx_max] = 0
                continue
                
        # Accept the center
        centers.append(X[idx_max])
        center_indices.append(idx_max)
        
        # Revise potentials
        dist_to_new_center_sq = np.sum((X - X[idx_max])**2, axis=1)
        potentials = potentials - P_max * np.exp(-4 * dist_to_new_center_sq / (rb ** 2))
        potentials[potentials < 0] = 0 # Ensure potentials don't drop below 0
        
    if len(centers) == 0:
        return np.array([]), np.zeros(n_samples, dtype=int) - 1
        
    centers = np.array(centers)
    
    # Assign each point to the closest center
    dist_to_centers = np.sqrt(np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(dist_to_centers, axis=1)
    
    return centers, labels

# ---------------------------------------------------------
# FUZZY C-MEANS (Bezdek 1981)
# ---------------------------------------------------------
def fuzzy_c_means(X, c, m=2.0, max_iter=100, tol=1e-4, init_centers=None):
    """
    Fuzzy C-Means from scratch.
    X: (n_samples, n_features)
    """
    n_samples, n_features = X.shape
    
    if c <= 0:
        return np.array([]), np.array([]), np.zeros(n_samples, dtype=int) - 1
        
    if init_centers is not None and len(init_centers) == c:
        v = init_centers.copy()
    else:
        indices = np.random.choice(n_samples, c, replace=False)
        v = X[indices].copy()
        
    u = np.zeros((n_samples, c))
    
    for _ in range(max_iter):
        dist = np.linalg.norm(X[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        
        inv_dist = dist ** (-2.0 / (m - 1))
        u_new = inv_dist / np.sum(inv_dist, axis=1)[:, np.newaxis]
        
        if np.linalg.norm(u_new - u) < tol:
            u = u_new
            break
        u = u_new
        
        um = u ** m
        v = np.dot(um.T, X) / np.sum(um, axis=0)[:, np.newaxis]
        
    labels = np.argmax(u, axis=1)
    return v, u, labels

# ---------------------------------------------------------
# PIPELINE UTILS
# ---------------------------------------------------------
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def reduce_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_red = pca.fit_transform(X)
    print(f"[INFO] Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.2%}")
    return X_red

def reduce_umap(X, n_neighbors=15, min_dist=0.1):
    n_neighbors = min(max(n_neighbors, 2), X.shape[0] - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    return reducer.fit_transform(X)

def plot_clusters(X_2d, labels, title, output_dir, filename, labels_reales=None):
    df_plot = pd.DataFrame({
        'Dim1': X_2d[:, 0],
        'Dim2': X_2d[:, 1],
        'Cluster': labels.astype(str)
    })
    
    style = None
    if labels_reales is not None:
        df_plot['Label_Real'] = labels_reales.astype(str)
        style = 'Label_Real'

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot, x='Dim1', y='Dim2',
        hue='Cluster', style=style, palette='tab10', s=100, alpha=0.85
    )
    plt.title(title)
    plt.legend(title='Clúster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def evaluate_silhouette(X, labels, method_name):
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    if len(unique_labels) > 1 and len(unique_labels) < len(labels[mask]):
        try:
            score = silhouette_score(X[mask], labels[mask])
            print(f"[INFO] Silhouette score ({method_name}): {score:.3f}")
        except ValueError:
            print(f"[WARNING] Silhouette falló para {method_name} (muy pocos items)")
    else:
        print(f"[WARNING] Silhouette no definido para {method_name}")

# ---------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Clustering Pipeline')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--n-clusters', type=int, default=None, help='Number of clusters for KMeans and FCM')
    parser.add_argument('--transpose', action='store_true', help='Transpose CSV (features as rows, samples as cols)')
    parser.add_argument('--ra', type=float, default=0.4, help='Subtractive clustering ra')
    parser.add_argument('--rb', type=float, default=0.6, help='Subtractive clustering rb')
    parser.add_argument('--eps-upper', type=float, default=0.5, help='Subtractive clustering eps-upper')
    parser.add_argument('--eps-lower', type=float, default=0.15, help='Subtractive clustering eps-lower')
    
    args = parser.parse_args()
    
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró el dataset en {csv_path}")
        return
        
    # Data Loading
    origin_df = pd.read_csv(csv_path)
    if args.transpose:
        # If transposed, the first column might be feature names, and headers might be sample names
        # Standard transpose
        df = origin_df.T
        # Reset index and clean up if necessary, depending on the format.
        # Simple transpose for general numeric CSVs:
        # Assuming all numeric for simplicity or dropping non-numeric
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    else:
        df = origin_df.copy()

    # Generic Preprocessing
    # Separate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    feature_cols = [c for c in numeric_cols if c not in ['anio', 'unidad', 'label']]
    if not feature_cols:
        feature_cols = numeric_cols # Fallback
        
    print(f"[INFO] Datos cargados o transpuestos: {df.shape[0]} muestras, {len(feature_cols)} atributos.")
    
    # Imputation
    df_processed = df.copy()
    for col in feature_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
    X_raw = df_processed[feature_cols].values
    
    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Real labels if available
    labels_reales = df_processed['label'].values if 'label' in df_processed.columns and not args.transpose else None
    
    # Output directory
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(get_script_dir(), 'results', 'clustering', base_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Resultados se guardarán en: {out_dir}")
    
    # Reductions
    X_pca = reduce_pca(X)
    X_umap = reduce_umap(X)
    
    # 1. K-MEANS
    n_cl = args.n_clusters if args.n_clusters is not None else 3
    print(f"\n[INFO] Ejecutando KMeans (k={n_cl})...")
    kmeans = KMeans(n_clusters=n_cl, random_state=42, n_init='auto')
    labels_km = kmeans.fit_predict(X)
    plot_clusters(X_pca, labels_km, f'K-Means (k={n_cl}) + PCA', out_dir, 'kmeans_pca.png', labels_reales)
    evaluate_silhouette(X, labels_km, 'KMeans')

    # 2. SUBTRACTIVE CLUSTERING
    print("\n[INFO] Ejecutando Subtractive Clustering...")
    print(f"       ra={args.ra}, rb={args.rb}, eps_upper={args.eps_upper}, eps_lower={args.eps_lower}")
    centers_sub, labels_sub = subtractive_clustering(
        X, ra=args.ra, rb=args.rb, eps_upper=args.eps_upper, eps_lower=args.eps_lower
    )
    n_sub_clusters = len(centers_sub)
    print(f"       -> {n_sub_clusters} clústeres encontrados.")
    plot_clusters(X_pca, labels_sub, f'Subtractive (C={n_sub_clusters}) + PCA', out_dir, 'subtractive_pca.png', labels_reales)
    evaluate_silhouette(X, labels_sub, 'Subtractive')

    # 3. FUZZY C-MEANS
    print(f"\n[INFO] Ejecutando Fuzzy C-Means (C={n_cl})...")
    # Initialize FCM with subtractive centers if k matches exactly or no n_clusters specified
    init_c = None
    if n_sub_clusters == n_cl:
        print("       -> Inicializando con los centros del Subtractive Clustering.")
        init_c = centers_sub
    
    v_fcm, u_fcm, labels_fcm = fuzzy_c_means(X, c=n_cl, init_centers=init_c)
    plot_clusters(X_pca, labels_fcm, f'FCM (C={n_cl}) + PCA', out_dir, 'fcm_pca.png', labels_reales)
    evaluate_silhouette(X, labels_fcm, 'FCM')

    # 4. DBSCAN
    print("\n[INFO] Ejecutando DBSCAN...")
    labels_db = DBSCAN(eps=2.5, min_samples=3).fit_predict(X)
    plot_clusters(X_pca, labels_db, 'DBSCAN + PCA', out_dir, 'dbscan_pca.png', labels_reales)
    evaluate_silhouette(X, labels_db, 'DBSCAN')

    # 5. HDBSCAN
    print("\n[INFO] Ejecutando HDBSCAN...")
    labels_hdb = hdbscan.HDBSCAN(min_cluster_size=4).fit_predict(X)
    plot_clusters(X_pca, labels_hdb, 'HDBSCAN + PCA', out_dir, 'hdbscan_pca.png', labels_reales)
    evaluate_silhouette(X, labels_hdb, 'HDBSCAN')

    print("\n" + "="*60)
    print("Pipeline de Clustering: Terminado con éxito.")
    print("="*60)

if __name__ == '__main__':
    main()
