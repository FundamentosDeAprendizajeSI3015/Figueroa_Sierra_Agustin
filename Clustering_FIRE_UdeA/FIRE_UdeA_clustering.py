"""
FIRE-UdeA — Agrupamiento (Clustering) con KMeans y DBSCAN
==========================================================
Adaptación del procedimiento del notebook ejAgrupamiento_kmeans_dbscan.ipynb
al dataset financiero y problema de riesgo de la Universidad de Antioquia.

Uso:
    python FIRE_UdeA_clustering.py

Salida: results/clustering/ con gráficas PNG y reportes CSV/TXT.

Universidad de Antioquia — Clase Fundamentos de Aprendizaje Automático
Fecha: 2026-03-17
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN, KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# CONFIGURACIÓN
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_sintetico_FIRE_UdeA_realista.csv')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'clustering')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reproducibilidad
RANDOM_STATE = 42

# Configuración visual
plt.rc('font', family='serif', size=12)
sns.set_theme(style='whitegrid', font_scale=1.1)

# Colores para clusters y labels
CLUSTER_CMAP = plt.cm.Set2
LABEL_COLORS = {0: '#4CAF50', 1: '#F44336'}

REPORT_LINES: list[str] = []


def log(msg: str) -> None:
    """Imprime y guarda en el reporte final."""
    print(msg)
    REPORT_LINES.append(msg)


# ============================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV y realiza validaciones básicas."""
    df = pd.read_csv(path)
    log("=" * 70)
    log("1. CARGA Y PREPARACIÓN DE DATOS")
    log("=" * 70)
    log(f"\nShape: {df.shape[0]} filas × {df.shape[1]} columnas")
    log(f"Años: {sorted(df['anio'].unique())} ({df['anio'].nunique()} únicos)")
    log(f"Unidades: {sorted(df['unidad'].unique())} ({df['unidad'].nunique()} únicas)")

    vc = df['label'].value_counts().sort_index()
    log(f"\nDistribución de etiqueta supervisada (label):")
    log(f"  label=0 (sano):   {vc.get(0, 0)} ({vc.get(0, 0)/len(df)*100:.1f}%)")
    log(f"  label=1 (riesgo): {vc.get(1, 0)} ({vc.get(1, 0)/len(df)*100:.1f}%)")
    return df


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    """Retorna los nombres de features numéricas excluyendo identificadores y target."""
    exclude = {'anio', 'label'}
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]


def prepare_data(df: pd.DataFrame, feature_cols: list[str]):
    """
    Prepara la matriz de datos:
      - Imputa valores faltantes con la mediana de cada columna.
      - Retorna la matriz X y un DataFrame auxiliar para trazabilidad.
    """
    df_work = df.copy()
    n_missing = df_work[feature_cols].isnull().sum().sum()
    log(f"\nValores faltantes en features: {n_missing}")

    for col in feature_cols:
        median_val = df_work[col].median()
        df_work[col] = df_work[col].fillna(median_val)

    X = df_work[feature_cols].values
    log(f"Matriz de datos lista: {X.shape[0]} muestras × {X.shape[1]} features")
    return X, df_work


# ============================================================
# 2. PIPELINE DE PRE-PROCESAMIENTO (igual estilo del notebook)
# ============================================================
def build_preprocessor(n_features: int) -> ColumnTransformer:
    """
    Construye un ColumnTransformer con StandardScaler para todas las
    features numéricas, replicando la misma arquitectura del notebook.
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, np.arange(n_features)),
        ],
    )
    return preprocessor


# ============================================================
# 3. K-MEANS CON K=2 (adaptación del Dataset 1 del notebook)
# ============================================================
def run_kmeans_k2(X: np.ndarray, preprocessor: ColumnTransformer):
    """
    Aplica KMeans con K=2 usando el pipeline de preprocesamiento.
    Retorna el pipeline entrenado y las etiquetas de cluster.
    """
    log("\n" + "=" * 70)
    log("2. K-MEANS CON K=2 (análogo al Dataset 1 del notebook)")
    log("=" * 70)

    clu_kmeans = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clustering", KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)),
        ]
    )
    clu_kmeans.fit(X)

    labels = clu_kmeans["clustering"].labels_
    inertia = clu_kmeans["clustering"].inertia_

    log(f"\n  K = 2")
    log(f"  Inercia: {inertia:.2f}")
    log(f"  Distribución de clusters: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if len(np.unique(labels)) > 1:
        X_scaled = clu_kmeans["preprocessor"].transform(X)
        sil = silhouette_score(X_scaled, labels)
        log(f"  Silhouette Score: {sil:.4f}")
    else:
        log("  Silhouette Score: N/A (un solo cluster)")

    return clu_kmeans, labels


# ============================================================
# 4. MÉTODO DEL CODO (adaptación del notebook)
# ============================================================
def elbow_method(X: np.ndarray, preprocessor: ColumnTransformer,
                 k_range: range = range(1, 11)):
    """
    Calcula la inercia para cada K y genera la gráfica del codo.
    Retorna el K óptimo estimado y las inercias.
    """
    log("\n" + "=" * 70)
    log("3. MÉTODO DEL CODO PARA DETERMINAR K ÓPTIMO")
    log("=" * 70)

    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in k_range:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clustering", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)),
            ]
        )
        pipeline.fit(X)
        inertia = pipeline["clustering"].inertia_
        inertias.append(inertia)

        if k >= 2:
            X_scaled = pipeline["preprocessor"].transform(X)
            sil = silhouette_score(X_scaled, pipeline["clustering"].labels_)
            silhouettes.append(sil)
            log(f"  K={k:2d}  |  Inercia={inertia:12.2f}  |  Silhouette={sil:.4f}")
        else:
            silhouettes.append(np.nan)
            log(f"  K={k:2d}  |  Inercia={inertia:12.2f}  |  Silhouette=N/A")

    # Estimar K óptimo por máximo silhouette
    valid_sil = [(k, s) for k, s in zip(k_range, silhouettes) if not np.isnan(s)]
    if valid_sil:
        best_k, best_sil = max(valid_sil, key=lambda x: x[1])
        log(f"\n  K óptimo (máximo Silhouette): K={best_k} (Silhouette={best_sil:.4f})")
    else:
        best_k = 2
        log(f"\n  K óptimo por defecto: K=2")

    # --- Gráfica del codo ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(list(k_range), inertias, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax.set_xlabel('K (número de clusters)')
    ax.set_ylabel('Inercia')
    ax.set_title('Método del Codo — Inercia', fontsize=13, fontweight='bold')
    ax.set_xticks(list(k_range))

    ax = axes[1]
    valid_k = [k for k, s in zip(k_range, silhouettes) if not np.isnan(s)]
    valid_s = [s for s in silhouettes if not np.isnan(s)]
    ax.plot(valid_k, valid_s, 's-', color='#FF9800', linewidth=2, markersize=8)
    ax.set_xlabel('K (número de clusters)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score por K', fontsize=13, fontweight='bold')
    ax.set_xticks(valid_k)

    plt.suptitle('FIRE-UdeA — Selección de K para KMeans',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '01_elbow_method.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ Gráfica guardada: 01_elbow_method.png")

    return best_k, inertias, silhouettes


# ============================================================
# 5. K-MEANS CON K ÓPTIMO
# ============================================================
def run_kmeans_optimal(X: np.ndarray, preprocessor: ColumnTransformer,
                       best_k: int):
    """
    Ejecuta KMeans con el K óptimo encontrado por el método del codo.
    """
    log("\n" + "=" * 70)
    log(f"4. K-MEANS CON K ÓPTIMO = {best_k}")
    log("=" * 70)

    clu_kmeans = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clustering", KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)),
        ]
    )
    clu_kmeans.fit(X)

    labels = clu_kmeans["clustering"].labels_
    inertia = clu_kmeans["clustering"].inertia_

    log(f"\n  K = {best_k}")
    log(f"  Inercia: {inertia:.2f}")
    log(f"  Distribución de clusters: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if len(np.unique(labels)) > 1:
        X_scaled = clu_kmeans["preprocessor"].transform(X)
        sil = silhouette_score(X_scaled, labels)
        log(f"  Silhouette Score: {sil:.4f}")

    return clu_kmeans, labels


# ============================================================
# 6. DBSCAN (adaptación de los Datasets 2 y 3 del notebook)
# ============================================================
def find_optimal_eps(X_scaled: np.ndarray, min_samples: int) -> float:
    """
    Estima un buen valor de eps usando el gráfico de distancias k-NN.
    Retorna el eps estimado como el punto de inflexión.
    """
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    # Heurística: buscar el punto de máxima curvatura
    # Usar segunda derivada discreta
    if len(k_distances) > 4:
        diffs = np.diff(k_distances)
        diffs2 = np.diff(diffs)
        if len(diffs2) > 0:
            knee_idx = np.argmax(diffs2) + 2
            eps_est = float(k_distances[min(knee_idx, len(k_distances) - 1)])
        else:
            eps_est = float(np.median(k_distances))
    else:
        eps_est = float(np.median(k_distances))

    return max(eps_est, 0.1)  # mínimo razonable


def run_dbscan(X: np.ndarray, preprocessor: ColumnTransformer):
    """
    Aplica DBSCAN con búsqueda automática de hiperparámetros.
    Retorna las etiquetas de cluster y el pipeline.
    """
    log("\n" + "=" * 70)
    log("5. DBSCAN (análogo a los Datasets 2 y 3 del notebook)")
    log("=" * 70)

    # Primero escalar los datos
    X_scaled = preprocessor.fit_transform(X)

    # Búsqueda de hiperparámetros
    best_sil = -1.0
    best_labels = None
    best_eps = 0.5
    best_min_samples = 5
    results: list[dict] = []

    for min_samples in [3, 4, 5, 7, 10]:
        # Estimar eps con k-NN
        eps_auto = find_optimal_eps(X_scaled, min_samples)

        # Probar también variaciones alrededor del eps estimado
        for eps_factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            eps = eps_auto * eps_factor

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            if n_clusters >= 2:
                # Silhouette solo sobre puntos no-ruido
                mask = labels != -1
                if mask.sum() >= 2 and len(np.unique(labels[mask])) >= 2:
                    sil = silhouette_score(X_scaled[mask], labels[mask])
                else:
                    sil = -1.0
            else:
                sil = -1.0

            results.append({
                'eps': round(eps, 4),
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': round(sil, 4),
            })

            if sil > best_sil:
                best_sil = sil
                best_labels = labels.copy()
                best_eps = eps
                best_min_samples = min_samples

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'dbscan_search.csv'), index=False)

    log(f"\n  Búsqueda de hiperparámetros completada ({len(results)} configuraciones)")
    log(f"  Mejores hiperparámetros: eps={best_eps:.4f}, min_samples={best_min_samples}")
    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise = (best_labels == -1).sum()
    log(f"  Clusters encontrados: {n_clusters}")
    log(f"  Puntos de ruido: {n_noise} ({n_noise/len(best_labels)*100:.1f}%)")
    log(f"  Silhouette Score: {best_sil:.4f}")
    log(f"\n  Top 10 configuraciones:")
    top10 = results_df.sort_values('silhouette', ascending=False).head(10)
    log(top10.to_string(index=False))

    # Gráfica k-NN distances para el min_samples óptimo
    nn = NearestNeighbors(n_neighbors=best_min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(k_distances)), k_distances, 'o-', color='#9C27B0',
            markersize=4, linewidth=1.5)
    ax.axhline(y=best_eps, color='#F44336', linestyle='--', linewidth=2,
               label=f'eps óptimo = {best_eps:.3f}')
    ax.set_xlabel('Índice (ordenado)')
    ax.set_ylabel(f'Distancia al {best_min_samples}-ésimo vecino más cercano')
    ax.set_title(f'Gráfico k-NN Distances (k={best_min_samples})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '02_knn_distances.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ Gráfica guardada: 02_knn_distances.png")

    return best_labels, best_eps, best_min_samples


# ============================================================
# 7. VISUALIZACIÓN DE CLUSTERS (PCA 2D)
# ============================================================
def visualize_clusters(X: np.ndarray, df_work: pd.DataFrame,
                       labels_k2: np.ndarray,
                       labels_kopt: np.ndarray, best_k: int,
                       labels_dbscan: np.ndarray,
                       feature_cols: list[str]):
    """
    Genera scatter plots 2D usando PCA para visualizar los clusters.
    """
    log("\n" + "=" * 70)
    log("6. VISUALIZACIÓN DE CLUSTERS (proyección PCA 2D)")
    log("=" * 70)

    # Escalar y reducir a 2D con PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_
    log(f"\n  Varianza explicada por PCA: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}")
    log(f"  Total: {sum(var_explained):.3f}")

    y_true = df_work['label'].values

    # --- 4 paneles: label real, K=2, K óptimo, DBSCAN ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    datasets = [
        (y_true, 'Etiqueta Supervisada (label)', False),
        (labels_k2, 'KMeans K=2', False),
        (labels_kopt, f'KMeans K={best_k} (óptimo)', False),
        (labels_dbscan, 'DBSCAN', True),
    ]

    for idx, (labels, title, is_dbscan) in enumerate(datasets):
        ax = axes[idx // 2][idx % 2]

        unique_labels = sorted(set(labels))

        for lbl in unique_labels:
            mask = labels == lbl
            if is_dbscan and lbl == -1:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c='gray', marker='x', s=40, alpha=0.5,
                           label='Ruido (-1)')
            elif idx == 0:
                color = LABEL_COLORS.get(lbl, '#999999')
                label_text = f"{'Sano' if lbl == 0 else 'Riesgo'} ({lbl})"
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=color, s=60, alpha=0.7, edgecolors='white',
                           linewidth=0.5, label=label_text)
            else:
                color = CLUSTER_CMAP(lbl / max(1, max(unique_labels)))
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[color], s=60, alpha=0.7, edgecolors='white',
                           linewidth=0.5, label=f'Cluster {lbl}')

        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')

    plt.suptitle('FIRE-UdeA — Comparación de Agrupamientos (PCA 2D)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '03_clusters_pca.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ Gráfica guardada: 03_clusters_pca.png")

    # --- Scatter usando las 2 features originales más importantes ---
    # (por varianza en el espacio escalado)
    stds = np.std(X_scaled, axis=0)
    top2_idx = np.argsort(stds)[::-1][:2]
    f1, f2 = feature_cols[top2_idx[0]], feature_cols[top2_idx[1]]
    log(f"\n  Top 2 features por varianza: {f1}, {f2}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (labels, title, is_dbscan) in enumerate(datasets):
        ax = axes[idx // 2][idx % 2]
        unique_labels = sorted(set(labels))

        for lbl in unique_labels:
            mask = labels == lbl
            if is_dbscan and lbl == -1:
                ax.scatter(X[mask, top2_idx[0]], X[mask, top2_idx[1]],
                           c='gray', marker='x', s=40, alpha=0.5,
                           label='Ruido (-1)')
            elif idx == 0:
                color = LABEL_COLORS.get(lbl, '#999999')
                label_text = f"{'Sano' if lbl == 0 else 'Riesgo'} ({lbl})"
                ax.scatter(X[mask, top2_idx[0]], X[mask, top2_idx[1]],
                           c=color, s=60, alpha=0.7, edgecolors='white',
                           linewidth=0.5, label=label_text)
            else:
                color = CLUSTER_CMAP(lbl / max(1, max(unique_labels)))
                ax.scatter(X[mask, top2_idx[0]], X[mask, top2_idx[1]],
                           c=[color], s=60, alpha=0.7, edgecolors='white',
                           linewidth=0.5, label=f'Cluster {lbl}')

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')

    plt.suptitle(f'FIRE-UdeA — Agrupamientos en espacio {f1} × {f2}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '04_clusters_features.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ Gráfica guardada: 04_clusters_features.png")


# ============================================================
# 8. COMPARACIÓN CON ETIQUETAS SUPERVISADAS
# ============================================================
def compare_with_labels(df_work: pd.DataFrame,
                        labels_k2: np.ndarray,
                        labels_kopt: np.ndarray, best_k: int,
                        labels_dbscan: np.ndarray):
    """
    Compara los clusters encontrados con las etiquetas supervisadas
    usando métricas de validación externa.
    """
    log("\n" + "=" * 70)
    log("7. COMPARACIÓN CON ETIQUETAS SUPERVISADAS")
    log("=" * 70)

    y_true = df_work['label'].values

    methods = [
        ('KMeans K=2', labels_k2),
        (f'KMeans K={best_k}', labels_kopt),
        ('DBSCAN', labels_dbscan),
    ]

    comparison_rows: list[dict] = []

    for name, labels in methods:
        # Filtrar ruido de DBSCAN para métricas
        if name == 'DBSCAN':
            mask = labels != -1
            y_t = y_true[mask]
            l = labels[mask]
            n_noise = (~mask).sum()
        else:
            y_t = y_true
            l = labels
            n_noise = 0

        if len(np.unique(l)) < 2 or len(y_t) < 2:
            log(f"\n  {name}: No se pueden calcular métricas (clusters insuficientes)")
            comparison_rows.append({
                'method': name,
                'ari': np.nan,
                'nmi': np.nan,
                'n_clusters': len(np.unique(l)),
                'n_noise': n_noise,
            })
            continue

        ari = adjusted_rand_score(y_t, l)
        nmi = normalized_mutual_info_score(y_t, l)

        log(f"\n  {name}:")
        log(f"    Adjusted Rand Index (ARI):            {ari:.4f}")
        log(f"    Normalized Mutual Information (NMI):   {nmi:.4f}")
        log(f"    Clusters: {len(np.unique(l))}, Ruido: {n_noise}")

        comparison_rows.append({
            'method': name,
            'ari': round(ari, 4),
            'nmi': round(nmi, 4),
            'n_clusters': len(np.unique(l)),
            'n_noise': n_noise,
        })

        # Tabla de contingencia
        log(f"\n    Tabla de contingencia (filas=cluster, columnas=label):")
        ct = pd.crosstab(
            pd.Series(l, name='cluster'),
            pd.Series(y_t, name='label'),
        )
        log(f"    {ct.to_string()}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_metrics.csv'), index=False)
    log("\n  ✓ Métricas guardadas: comparison_metrics.csv")

    # --- Gráfica de barras comparativa ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    method_names = comparison_df['method'].values
    x_pos = np.arange(len(method_names))

    ax = axes[0]
    bars = ax.bar(x_pos, comparison_df['ari'].fillna(0), color='#42A5F5',
                  edgecolor='white', width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('ARI (concordancia con label)', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, comparison_df['ari'].fillna(0)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)

    ax = axes[1]
    bars = ax.bar(x_pos, comparison_df['nmi'].fillna(0), color='#FF7043',
                  edgecolor='white', width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Normalized Mutual Information')
    ax.set_title('NMI (información compartida con label)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, comparison_df['nmi'].fillna(0)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)

    plt.suptitle('FIRE-UdeA — Métricas de Validación Externa',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '05_comparison_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ Gráfica guardada: 05_comparison_metrics.png")


# ============================================================
# 9. ANÁLISIS DE PERFILES DE CLUSTERS
# ============================================================
def cluster_profiles(df_work: pd.DataFrame, feature_cols: list[str],
                     labels_kopt: np.ndarray, best_k: int):
    """
    Genera un perfil estadístico de cada cluster (media de cada feature).
    """
    log("\n" + "=" * 70)
    log(f"8. PERFILES DE CLUSTERS (KMeans K={best_k})")
    log("=" * 70)

    df_prof = df_work.copy()
    df_prof['cluster'] = labels_kopt

    profile = df_prof.groupby('cluster')[feature_cols].mean()
    log(f"\n  Media de cada feature por cluster:")
    log(profile.T.to_string())

    profile.T.to_csv(os.path.join(RESULTS_DIR, 'cluster_profiles.csv'))
    log("  ✓ Perfiles guardados: cluster_profiles.csv")

    # Heatmap de perfiles normalizados
    scaler = StandardScaler()
    profile_scaled = pd.DataFrame(
        scaler.fit_transform(profile.T).T,
        columns=feature_cols,
        index=profile.index,
    )

    fig, ax = plt.subplots(figsize=(14, max(4, best_k * 1.5)))
    sns.heatmap(profile_scaled, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, linewidths=0.5, ax=ax, annot_kws={'size': 8})
    ax.set_title(f'Perfiles de Clusters (KMeans K={best_k}, escalados)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Cluster')
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '06_cluster_profiles.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ Gráfica guardada: 06_cluster_profiles.png")

    # Composición de cada cluster por unidad y label
    log(f"\n  Composición de clusters por unidad:")
    comp_unit = pd.crosstab(df_prof['cluster'], df_prof['unidad'])
    log(comp_unit.to_string())

    log(f"\n  Composición de clusters por label supervisada:")
    comp_label = pd.crosstab(df_prof['cluster'], df_prof['label'])
    log(comp_label.to_string())


# ============================================================
# 10. RESUMEN Y CONCLUSIONES
# ============================================================
def summary_and_conclusions(best_k: int):
    """Resumen final del análisis de agrupamiento."""
    log("\n" + "=" * 70)
    log("9. RESUMEN Y CONCLUSIONES")
    log("=" * 70)
    log(f"""
  El análisis de agrupamiento aplicó dos algoritmos sobre el dataset
  financiero FIRE-UdeA:

  ■ KMeans: algoritmo basado en centroides, probado con K=2 (análogo
    al ejercicio del notebook) y con K óptimo = {best_k} determinado
    por el método del codo y Silhouette Score.

  ■ DBSCAN: algoritmo basado en densidad con búsqueda automática de
    hiperparámetros (eps y min_samples), capaz de detectar puntos de
    ruido (outliers financieros).

  Se compararon los agrupamientos no supervisados contra la etiqueta
  supervisada (label: sano/riesgo) usando ARI y NMI para evaluar la
  concordancia.

  Todos los resultados se guardaron en: {RESULTS_DIR}
""")


# ============================================================
# MAIN
# ============================================================
def main():
    """Pipeline principal de agrupamiento FIRE-UdeA."""
    log("=" * 70)
    log("FIRE-UdeA — AGRUPAMIENTO (CLUSTERING) CON KMEANS Y DBSCAN")
    log(f"Adaptación del notebook ejAgrupamiento_kmeans_dbscan.ipynb")
    log("=" * 70)

    # 1. Cargar datos
    df = load_data(DATA_PATH)
    feature_cols = get_numeric_features(df)
    log(f"\nFeatures numéricas ({len(feature_cols)}): {feature_cols}")
    X, df_work = prepare_data(df, feature_cols)

    # 2. Construir preprocessor (misma arquitectura del notebook)
    preprocessor = build_preprocessor(X.shape[1])

    # 3. KMeans K=2
    _, labels_k2 = run_kmeans_k2(X, preprocessor)

    # 4. Método del codo
    preprocessor_elbow = build_preprocessor(X.shape[1])
    best_k, _, _ = elbow_method(X, preprocessor_elbow)

    # 5. KMeans K óptimo
    preprocessor_opt = build_preprocessor(X.shape[1])
    _, labels_kopt = run_kmeans_optimal(X, preprocessor_opt, best_k)

    # 6. DBSCAN
    preprocessor_dbscan = build_preprocessor(X.shape[1])
    labels_dbscan, _, _ = run_dbscan(X, preprocessor_dbscan)

    # 7. Visualización
    visualize_clusters(X, df_work, labels_k2, labels_kopt, best_k,
                       labels_dbscan, feature_cols)

    # 8. Comparación con etiquetas supervisadas
    compare_with_labels(df_work, labels_k2, labels_kopt, best_k, labels_dbscan)

    # 9. Perfiles de clusters
    cluster_profiles(df_work, feature_cols, labels_kopt, best_k)

    # 10. Resumen
    summary_and_conclusions(best_k)

    # Guardar reporte de texto
    report_path = os.path.join(RESULTS_DIR, 'clustering_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(REPORT_LINES))
    log(f"\n✓ Reporte completo guardado: {report_path}")
    log("=" * 70)
    log("AGRUPAMIENTO COMPLETADO")
    log("=" * 70)


if __name__ == '__main__':
    main()
