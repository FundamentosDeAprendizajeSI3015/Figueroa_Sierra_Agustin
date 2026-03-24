# ==========================================================
# Clustering Pipeline: Hierarchical, DBSCAN, HDBSCAN & UMAP
# Adapted for FIRE_UdeA Project
# ==========================================================

import os
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

# Import project specific feature engineering
try:
    from FIRE_UdeA_model_v2 import engineer_features, get_feature_columns
except ImportError:
    print("[WARNING] Could not import engineer_features from FIRE_UdeA_model_v2. Using base features.")
    engineer_features = None
    get_feature_columns = None

sns.set(style="whitegrid", context="talk")

# -----------------------------
# 1. CONFIG & PATHS
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    {
        "path": os.path.join(SCRIPT_DIR, 'dataset_sintetico_FIRE_UdeA_realista.csv'),
        "out_dir": os.path.join(SCRIPT_DIR, 'results', 'clustering', 'realista')
    },
    {
        "path": os.path.join(SCRIPT_DIR, 'legacy_files', 'dataset_sintetico_FIRE_UdeA.csv'),
        "out_dir": os.path.join(SCRIPT_DIR, 'results', 'clustering', 'legacy')
    }
]

# -----------------------------
# 2. CARGA DE DATOS
# -----------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Carga datos desde un archivo CSV."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Datos cargados: {df.shape[0]} observaciones, {df.shape[1]} variables")
    return df

# -----------------------------
# 3. PREPROCESAMIENTO ADAPTADO
# -----------------------------
def preprocess_data(df: pd.DataFrame):
    """
    Escalamiento estándar e imputación de valores faltantes.
    Adaptado para usar el feature engineering del proyecto FIRE_UdeA.
    """
    df_processed = df.copy()
    
    # Aplicar feature engineering del proyecto si está disponible
    if engineer_features is not None:
        try:
            df_processed = engineer_features(df_processed)
            feature_cols = get_feature_columns(df_processed)
        except Exception as e:
            print(f"[WARNING] Falló el feature engineering: {e}. Usando variables base.")
            feature_cols = [c for c in df_processed.columns if c not in ['anio', 'unidad', 'label']]
    else:
        feature_cols = [c for c in df_processed.columns if c not in ['anio', 'unidad', 'label']]
        
    print(f"[INFO] Variables a usar para clustering: {feature_cols}")
    
    # Imputación simple por la mediana para clustering no supervisado
    for col in feature_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    X = df_processed[feature_cols].values
    
    # Escalamiento estándar (fundamental para distancias en clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_processed, feature_cols

# -----------------------------
# 4. CLUSTERING JERÁRQUICO
# -----------------------------
def hierarchical_clustering(X: np.ndarray, method: str):
    return linkage(X, method=method)

def plot_dendrogram(Z, method: str, output_dir: str, truncate_level: int = 40):
    plt.figure(figsize=(14, 6))
    dendrogram(Z, truncate_mode='level', p=truncate_level)
    plt.title(f"Dendrograma jerárquico – {method.upper()} (FIRE_UdeA)")
    plt.xlabel("Observaciones")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'dendrograma_{method}.png'))
    plt.close()

# -----------------------------
# 5. DBSCAN
# -----------------------------
def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

# -----------------------------
# 6. HDBSCAN
# -----------------------------
def run_hdbscan(X: np.ndarray, min_cluster_size: int = 5):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return labels, model

# -----------------------------
# 7. REDUCCIÓN DE DIMENSIÓN
# -----------------------------
def reduce_pca(X: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components, random_state=42)
    X_red = pca.fit_transform(X)
    print(f"[INFO] Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.2%}")
    return X_red

def reduce_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1):
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    if n_neighbors < 2: n_neighbors = 2
        
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42
    )
    return reducer.fit_transform(X)

# -----------------------------
# 8. VISUALIZACIÓN
# -----------------------------
def plot_clusters(X_2d: np.ndarray, labels: np.ndarray, df: pd.DataFrame, title: str, output_dir: str, filename: str):
    df_plot = pd.DataFrame({
        'Dim1': X_2d[:, 0],
        'Dim2': X_2d[:, 1],
        'Cluster': labels.astype(str),
        'Unidad': df['unidad'] if 'unidad' in df.columns else 'N/A',
        'Label_Real': df['label'] if 'label' in df.columns else 'N/A'
    })

    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        data=df_plot,
        x='Dim1',
        y='Dim2',
        hue='Cluster',
        style='Label_Real',  # Mostramos el label real de riesgo en el estilo
        palette='tab10',
        s=100,
        alpha=0.85
    )
    plt.title(f"{title} (FIRE_UdeA)")
    plt.legend(title='Clúster / Riesgo Real', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# -----------------------------
# 9. EVALUACIÓN
# -----------------------------
def silhouette(X: np.ndarray, labels: np.ndarray, method_name: str):
    mask = labels != -1
    if len(np.unique(labels[mask])) > 1:
        score = silhouette_score(X[mask], labels[mask])
        print(f"[INFO] Silhouette score ({method_name}): {score:.3f}")
    else:
        print(f"[WARNING] Silhouette no definido para {method_name} (muy pocos clústeres válidos o todo ruido)")

# -----------------------------
# 10. PIPELINE PRINCIPAL POR DATASET
# -----------------------------
def run_pipeline_for_dataset(csv_path: str, output_dir: str):
    print("="*60)
    print(f"Dataset:  {os.path.basename(csv_path)}")
    print(f"Salida:   {output_dir}")
    print("="*60)
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró el dataset en {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = load_data(csv_path)
    X, df_processed, feature_cols = preprocess_data(df)
    
    # Reducciones de dimensión para visualización
    print("\n[INFO] Ejecutando reducciones de dimensión (PCA y UMAP)...")
    X_pca = reduce_pca(X)
    X_umap = reduce_umap(X, n_neighbors=10) # 10 neighbors por ser dataset con pocos datos

    # -----------------------------------------------------
    # CLASIFICACIÓN REAL (Unos y Ceros del dataset original)
    # -----------------------------------------------------
    print("\n[INFO] Graficando las clases reales (Sano=0, Riesgo=1)...")
    if 'label' in df_processed.columns:
        labels_reales = df_processed['label'].values
        plot_clusters(X_pca, labels_reales, df_processed, 'Clases Reales (Ground Truth) + PCA', output_dir, 'real_labels_pca.png')
        plot_clusters(X_umap, labels_reales, df_processed, 'Clases Reales (Ground Truth) + UMAP', output_dir, 'real_labels_umap.png')
    else:
        print("[WARNING] No se encontró la columna 'label' para graficar las clases reales.")

    # -----------------------------------------------------
    # K-MEANS (Forzando k=2 para predecir Unos y Ceros)
    # -----------------------------------------------------
    print("\n[INFO] Ejecutando KMeans (k=2 para buscar 2 grupos principales)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    labels_kmeans = kmeans.fit_predict(X)
    
    plot_clusters(X_pca, labels_kmeans, df_processed, 'KMeans (k=2) + PCA', output_dir, 'kmeans_pca.png')
    plot_clusters(X_umap, labels_kmeans, df_processed, 'KMeans (k=2) + UMAP', output_dir, 'kmeans_umap.png')
    silhouette(X, labels_kmeans, 'KMeans')

    # Clustering jerárquico
    print("\n[INFO] Ejecutando Clustering Jerárquico...")
    for method in ['single', 'complete', 'average', 'ward']:
        Z = hierarchical_clustering(X, method)
        plot_dendrogram(Z, method, output_dir)
    print(f"       -> Dendrogramas guardados en {output_dir}")

    # DBSCAN
    print("\n[INFO] Ejecutando DBSCAN...")
    eps_val = 2.5
    min_samples_val = 3
    labels_db = run_dbscan(X, eps=eps_val, min_samples=min_samples_val)
    
    plot_clusters(X_pca, labels_db, df_processed, f'DBSCAN (eps={eps_val}) + PCA', output_dir, 'dbscan_pca.png')
    plot_clusters(X_umap, labels_db, df_processed, f'DBSCAN (eps={eps_val}) + UMAP', output_dir, 'dbscan_umap.png')
    silhouette(X, labels_db, 'DBSCAN')

    # HDBSCAN
    print("\n[INFO] Ejecutando HDBSCAN...")
    labels_hdb, _ = run_hdbscan(X, min_cluster_size=4) # min_cluster=4 para identificar subgrupos pequeños
    plot_clusters(X_pca, labels_hdb, df_processed, 'HDBSCAN + PCA', output_dir, 'hdbscan_pca.png')
    plot_clusters(X_umap, labels_hdb, df_processed, 'HDBSCAN + UMAP', output_dir, 'hdbscan_umap.png')
    silhouette(X, labels_hdb, 'HDBSCAN')


def main():
    print("="*60)
    print("Iniciando Pipeline de Clustering FIRE_UdeA (Múltiples Datasets)")
    print("="*60)
    
    for dataset in DATASETS:
        run_pipeline_for_dataset(dataset["path"], dataset["out_dir"])
        print("\n")
        
    print(f"Flujos ejecutados correctamente. Revisa la carpeta {os.path.join(SCRIPT_DIR, 'results', 'clustering')}")


if __name__ == '__main__':
    main()
