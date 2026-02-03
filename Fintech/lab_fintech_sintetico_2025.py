
# This file contains preprocessing and EDA code for synthetic fintech data.
# =============================================================
# LAB FINTECH (SINTÉTICO 2025) — PREPROCESAMIENTO Y EDA
# Datos de entrada fijos para evitar errores de ruta/nombre.
# -------------------------------------------------------------
# Este script está listo para ejecutarse sin argumentos:
#   python lab_fintech_sintetico_2025.py
# 
# Archivos esperados en el mismo directorio:
#   - fintech_top_sintetico_2025.csv
#   - fintech_top_sintetico_dictionary.json
# Salidas (por defecto):
#   ./data_output_finanzas_sintetico/
#       ├─ fintech_train.parquet
#       ├─ fintech_test.parquet
#       ├─ processed_schema.json
#       └─ features_columns.txt
# =============================================================

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Imports adicionales para análisis avanzado y ML
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------
# Constantes de la práctica
# ---------------------------
DATA_CSV = 'fintech_top_sintetico_2025.csv'
DATA_DICT = 'fintech_top_sintetico_dictionary.json'
OUTDIR = Path('./data_output_finanzas_sintetico')
SPLIT_DATE = '2025-09-01'  # partición temporal por defecto

# Columnas esperadas por diseño del dataset sintético
DATE_COL = 'Month'
ID_COLS = ['Company']
CAT_COLS = ['Country', 'Region', 'Segment', 'Subsegment', 'IsPublic', 'Ticker']
NUM_COLS = [
    'Users_M','NewUsers_K','TPV_USD_B','TakeRate_pct','Revenue_USD_M',
    'ARPU_USD','Churn_pct','Marketing_Spend_USD_M','CAC_USD','CAC_Total_USD_M',
    'Close_USD','Private_Valuation_USD_B'
]
PRICE_COLS = ['Close_USD']  # para calcular retornos opcionales

# ---------------------------
# 0) Carga de diccionario
# ---------------------------
print("\n=== 0) Cargando diccionario de datos ===")
dict_path = Path(DATA_DICT)
if not dict_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_DICT}. Asegúrate de tener el archivo en la misma carpeta.")

with open(dict_path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)
print("Descripción:", data_dict.get('description', '(sin descripción)'))
print("Periodo:", data_dict.get('period', '(desconocido)'))

# ---------------------------
# 1) Carga del CSV
# ---------------------------
print("\n=== 1) Cargando CSV sintético ===")
csv_path = Path(DATA_CSV)
if not csv_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_CSV}. Asegúrate de tener el archivo en la misma carpeta.")

df = pd.read_csv(csv_path)
print("Shape:", df.shape)

# Parseo de fecha y orden temporal
if DATE_COL not in df.columns:
    raise KeyError(f"La columna de fecha '{DATE_COL}' no existe en el CSV.")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

print("Primeras filas:")
print(df.head(3))

# ---------------------------
# 2) EDA breve
# ---------------------------
print("\n=== 2) EDA rápido ===")
print("Info:")
print(df.info())
print("\nNulos por columna (top 15):")
print(df.isna().sum().sort_values(ascending=False).head(15))

# ---------------------------
# 3) Limpieza básica
# ---------------------------
print("\n=== 3) Limpieza ===")
# Imputación simple: numéricos con mediana, categóricos con marcador
for c in NUM_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

for c in CAT_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = df[c].fillna('__MISSING__')

# ---------------------------
# 4) Ingeniería ligera: retornos/log-retornos de precio
# ---------------------------
print("\n=== 4) Ingeniería de rasgos (retornos) ===")
if all([pc in df.columns for pc in PRICE_COLS]):
    for pc in PRICE_COLS:
        # Retornos por empresa y fecha
        df[pc + '_ret'] = (
            df.sort_values([ID_COLS[0], DATE_COL])
              .groupby(ID_COLS)[pc]
              .pct_change()
        )
        df[pc + '_logret'] = np.log1p(df[pc + '_ret'])
        # Imputar primeros NA en 0.0 para continuidad
        df[pc + '_ret'] = df[pc + '_ret'].fillna(0.0)
        df[pc + '_logret'] = df[pc + '_logret'].fillna(0.0)
else:
    print("[INFO] Columnas de precio no disponibles; se omite cálculo de retornos.")

# Actualizamos lista de numéricos tras ingeniería
extra_num = [c for c in [pc + '_ret' for pc in PRICE_COLS] + [pc + '_logret' for pc in PRICE_COLS] if c in df.columns]
NUM_USED = [c for c in NUM_COLS if c in df.columns] + extra_num

# ---------------------------
# 5) Separación X / y (sin y por defecto) + codificación
# ---------------------------
print("\n=== 5) Preparación de X: codificación one-hot y escalado ===")
# Quitamos identificadores y fecha de las variables predictoras
X = df.drop(columns=[DATE_COL] + ID_COLS, errors='ignore').copy()

# One-hot en categóricas
cat_in_X = [c for c in CAT_COLS if c in X.columns]
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)

# Partición temporal por defecto utilizando la fecha de corte
cutoff = pd.to_datetime(SPLIT_DATE)
idx_train = df[DATE_COL] < cutoff
idx_test = df[DATE_COL] >= cutoff

X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()

# Escalado de numéricos (solo columnas presentes en X)
num_in_X = [c for c in NUM_USED if c in X_train.columns]
scaler = StandardScaler()
if num_in_X:
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_test[num_in_X] = scaler.transform(X_test[num_in_X])
else:
    print("[INFO] No se encontraron columnas numéricas para escalar.")

print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)

# ---------------------------
# 6) Exportación
# ---------------------------
print("\n=== 6) Exportación ===")
OUTDIR.mkdir(parents=True, exist_ok=True)
train_path = OUTDIR / 'fintech_train.parquet'
test_path = OUTDIR / 'fintech_test.parquet'

# Guardamos sólo X (sin objetivo)
X_train.to_parquet(train_path, index=False)
X_test.to_parquet(test_path, index=False)

# Guardar esquema procesado
processed_schema = {
    'source_csv': str(csv_path.resolve()),
    'source_dict': str(dict_path.resolve()),
    'date_col': DATE_COL,
    'id_cols': ID_COLS,
    'categorical_cols_used': cat_in_X,
    'numeric_cols_used': num_in_X,
    'engineered_cols': extra_num,
    'split': {
        'type': 'time_split',
        'cutoff': SPLIT_DATE,
        'train_rows': int(idx_train.sum()),
        'test_rows': int(idx_test.sum()),
    },
    'X_train_shape': list(X_train.shape),
    'X_test_shape': list(X_test.shape),
    'notes': [
        'Dataset 100% SINTÉTICO con fines académicos; no refleja métricas reales.',
        'Evitar fuga de datos: el escalador se ajusta en TRAIN y se aplica a TEST.'
    ]
}

with open(OUTDIR / 'processed_schema.json', 'w', encoding='utf-8') as f:
    json.dump(processed_schema, f, ensure_ascii=False, indent=2)

# Lista de columnas finales para referencia de modelado
with open(OUTDIR / 'features_columns.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(X_train.columns))

print("\nArchivos exportados:")
print(" -", train_path)
print(" -", test_path)
print(" -", OUTDIR / 'processed_schema.json')
print(" -", OUTDIR / 'features_columns.txt')

# ---------------------------
# 7) Análisis avanzado de datos y ML
# ---------------------------
print("\n=== 7) Análisis avanzado y ML ===")

# Configuración de visualizaciones
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# a) Matriz de correlación para variables numéricas
print("Generando matriz de correlación...")
if num_in_X:
    corr_matrix = X_train[num_in_X].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación - Variables Numéricas', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Matriz de correlación guardada en correlation_matrix.png")

# b) Detección de outliers con boxplots (primeras 6 variables numéricas)
print("Generando boxplots para detección de outliers...")
for i, col in enumerate(num_in_X[:6]):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=X_train[col], color='skyblue')
    plt.title(f'Boxplot de {col} (Train)', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTDIR / f'boxplot_{col}.png', dpi=300, bbox_inches='tight')
    plt.close()
print(" - Boxplots guardados para las primeras 6 variables numéricas")

# c) Análisis de tendencias temporales (ejemplo con Revenue)
print("Generando gráfico de tendencias temporales...")
if 'Revenue_USD_M' in df.columns:
    plt.figure(figsize=(12, 6))
    for company in df['Company'].unique()[:5]:  # primeras 5 compañías
        subset = df[df['Company'] == company].sort_values(DATE_COL)
        plt.plot(subset[DATE_COL], subset['Revenue_USD_M'], label=company, marker='o', alpha=0.7)
    plt.title('Tendencias de Revenue por Compañía (Top 5)', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Revenue (USD M)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'revenue_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Gráfico de tendencias guardado en revenue_trends.png")

# d) Modelo de ML simple: Regresión Lineal para predecir Revenue
print("Entrenando modelo de Regresión Lineal...")
target_col = 'Revenue_USD_M'
if target_col in df.columns and target_col in X_train.columns:
    # Preparar y
    y_train = df.loc[idx_train, target_col].values
    y_test = df.loc[idx_test, target_col].values
    
    # Modelo
    model = LinearRegression()
    model.fit(X_train.drop(columns=['cluster'] if 'cluster' in X_train.columns else [], errors='ignore'), y_train)
    y_pred = model.predict(X_test.drop(columns=['cluster'] if 'cluster' in X_test.columns else [], errors='ignore'))
    
    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = model.score(X_test.drop(columns=['cluster'] if 'cluster' in X_test.columns else [], errors='ignore'), y_test)
    
    print(f" - MAE: {mae:.2f}")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - R²: {r2:.2f}")
    
    # Guardar predicciones
    pred_df = pd.DataFrame({
        'Company': df.loc[idx_test, 'Company'].values,
        'Month': df.loc[idx_test, DATE_COL].values,
        'y_true': y_test,
        'y_pred': y_pred,
        'error': y_test - y_pred
    })
    pred_df.to_csv(OUTDIR / 'predictions_revenue.csv', index=False)
    print(" - Predicciones guardadas en predictions_revenue.csv")
    
    # Gráfico de predicciones vs reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales', fontsize=12)
    plt.ylabel('Predicciones', fontsize=12)
    plt.title('Predicciones vs Valores Reales - Revenue', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Gráfico de predicciones guardado en predictions_scatter.png")

# e) Clustering con K-Means
print("Realizando clustering con K-Means...")
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
X_train_copy = X_train.drop(columns=['cluster'] if 'cluster' in X_train.columns else [], errors='ignore').copy()
clusters = kmeans.fit_predict(X_train_copy)
X_train_copy['cluster'] = clusters

# Guardar datos con clusters
X_train_copy.to_parquet(OUTDIR / 'fintech_train_with_clusters.parquet', index=False)
print(f" - Datos con clusters (k={n_clusters}) guardados en fintech_train_with_clusters.parquet")

# Visualización de clusters con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_copy.drop('cluster', axis=1))
plt.figure(figsize=(10, 8))
for cluster in range(n_clusters):
    mask = X_train_copy['cluster'] == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster}', alpha=0.7)
plt.title('Clusters visualizados con PCA', fontsize=16)
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTDIR / 'clusters_pca.png', dpi=300, bbox_inches='tight')
plt.close()
print(" - Visualización de clusters guardada en clusters_pca.png")

# f) Análisis de importancia de variables (coeficientes del modelo)
if 'model' in locals():
    feature_importance = pd.DataFrame({
        'feature': X_train.drop(columns=['cluster'] if 'cluster' in X_train.columns else [], errors='ignore').columns,
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Top 10 Variables Más Importantes (Regresión Lineal)', fontsize=16)
    plt.xlabel('Importancia Absoluta', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" - Importancia de variables guardada en feature_importance.png")

print("\n✔ Análisis avanzado completado. Archivos adicionales generados en", OUTDIR)

print("\n✔ Listo. Recuerda: este dataset es sintético para práctica académica.")
