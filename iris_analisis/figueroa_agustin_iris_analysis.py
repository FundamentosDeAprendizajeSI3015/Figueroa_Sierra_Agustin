# iris_analysis.py
# -*- coding: utf-8 -*-
"""
Análisis completo del dataset Iris en Python con mensajes de progreso.

Este script realiza un análisis exhaustivo del famoso dataset Iris, que contiene
medidas de 150 flores de iris de tres especies diferentes. El análisis incluye:

1. Carga y exploración inicial de datos
2. Ingeniería de características (creación de nuevas variables)
3. Reducción de dimensionalidad (PCA y t-SNE)
4. Entrenamiento y evaluación de múltiples modelos de machine learning
5. Optimización de hiperparámetros
6. Análisis de importancia de características
7. Análisis de clustering no supervisado
8. Guardado del mejor modelo y resultados

Autor: Agustín Figueroa
Fecha: Febrero 2026
"""

from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")  # Ignorar warnings para mantener la salida limpia

# Importaciones de bibliotecas estándar y de ciencia de datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importaciones de scikit-learn para machine learning
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump

# Configuración de semilla para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n[1/10] Cargando dataset Iris...")

# ==========================================
# 1) Carga de datos
# ==========================================
# Cargar el dataset Iris desde scikit-learn
# Este dataset contiene 150 muestras de flores con 4 características cada una
iris = datasets.load_iris()
# Crear DataFrame con las características (medidas de las flores)
X = pd.DataFrame(iris.data, columns=iris.feature_names)
# Crear Serie con las etiquetas (especies: 0=setosa, 1=versicolor, 2=virginica)
y = pd.Series(iris.target, name="target")
# Obtener los nombres de las clases
class_names = iris.target_names

# Limpiar nombres de columnas (remover unidades y espacios)
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]
# Crear DataFrame completo combinando características y especies
df = pd.concat([X, y.map({i: name for i, name in enumerate(class_names)}).rename("species")], axis=1)

os.makedirs("outputs", exist_ok=True)

# ==========================================
# 2) Exploración inicial
# ==========================================
print("\n[2/10] Exploración inicial de datos...")

# Mostrar dimensiones del dataset
print("\n=== Dimensiones ===")
print(df.shape)

# Mostrar primeras filas para ver la estructura
print("\n=== Primeras filas ===")
print(df.head())

# Estadísticas descriptivas de las variables
print("\n=== Descripción estadística ===")
print(df.describe(include="all"))

# Distribución de las clases (especies)
print("\n=== Clases ===")
print(df["species"].value_counts())

print("Generando gráficos exploratorios...")

# Matriz de correlación entre características
corr = df.drop(columns=["species"]).corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
plt.title("Matriz de correlación - Iris")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

# Pairplot para visualizar relaciones entre variables por especie
sns.pairplot(df, hue="species", corner=True)
plt.suptitle("Pairplot Iris", y=1.02)
plt.savefig("outputs/pairplot.png", dpi=150)
plt.close()

# Boxplots para comparar distribuciones por especie
plt.figure(figsize=(10,6))
for i, col in enumerate(X.columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x="species", y=col)
plt.suptitle("Distribuciones por especie")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("outputs/boxplots.png", dpi=150)
plt.close()

# ==========================================
# 3) Ingeniería de características
# ==========================================
print("\n[3/10] Ingeniería de características y reducción de dimensión...")

# Crear nuevas características derivadas de las originales
X_feat = X.copy()
# Ratio largo/ancho del sépalo (forma del sépalo)
X_feat["sepal_ratio"] = X["sepal_length"] / X["sepal_width"]
# Ratio largo/ancho del pétalo (forma del pétalo)
X_feat["petal_ratio"] = X["petal_length"] / X["petal_width"]
# Área del sépalo (aproximada)
X_feat["sepal_area"] = X["sepal_length"] * X["sepal_width"]
# Área del pétalo (aproximada)
X_feat["petal_area"] = X["petal_length"] * X["petal_width"]

print("Calculando PCA...")
# Reducción de dimensionalidad con PCA (Análisis de Componentes Principales)
# Reduce las 4 dimensiones a 2 para visualización
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

# Visualizar las clases en el espacio PCA
plt.figure(figsize=(6,5))
for i, name in enumerate(class_names):
    idx = y == i
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=name)
plt.title("PCA (2 componentes)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_2d.png", dpi=150)
plt.close()

print("Calculando t-SNE (esto puede tardar unos segundos)...")
# t-SNE (t-Distributed Stochastic Neighbor Embedding)
# Técnica avanzada de reducción de dimensionalidad que preserva relaciones locales
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X))
print("t-SNE completado.")

# Visualizar las clases en el espacio t-SNE
plt.figure(figsize=(6,5))
for i, name in enumerate(class_names):
    idx = y == i
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=name)
plt.title("t-SNE (2D)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/tsne_2d.png", dpi=150)
plt.close()

# ==========================================
# 4) Partición de datos
# ==========================================
print("\n[4/10] División train/test...")

# Dividir los datos en conjunto de entrenamiento (75%) y prueba (25%)
# Usar stratify para mantener la proporción de clases en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ==========================================
# 5) Modelos y pipelines
# ==========================================
print("\n[5/10] Definiendo modelos y ejecutando validación cruzada...")

# Definir diferentes modelos de machine learning en pipelines
# Los pipelines incluyen escalado automático donde es necesario
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),  # Escalado de características
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())  # K-Nearest Neighbors
    ]),
    "DecisionTree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))  # Árbol de decisión
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))  # Bosque aleatorio
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))  # Support Vector Machine con kernel RBF
    ])
}

# Configurar validación cruzada estratificada (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n=== Validación cruzada ===")
# Evaluar cada modelo usando validación cruzada
for name, pipe in models.items():
    print(f"Entrenando modelo: {name} ...")
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"{name:>15}: {scores.mean():.3f} ± {scores.std():.3f}")

# ==========================================
# 6) Ajuste de hiperparámetros
# ==========================================
print("\n[6/10] Ajuste de hiperparámetros (GridSearch)...")

# Definir grilla de parámetros para SVM
param_grid_svm = {
    "clf__C": [1, 10],  # Parámetro de regularización
    "clf__gamma": ["scale", 0.1]  # Parámetro del kernel
}

print("Iniciando GridSearch para SVM...")
# Búsqueda de hiperparámetros óptimos para SVM
svm_grid = GridSearchCV(
    models["SVM_RBF"],
    param_grid_svm,
    cv=cv,
    scoring="accuracy",
    n_jobs=1,  # Usar 1 job para evitar problemas en algunos entornos
    verbose=2
)
svm_grid.fit(X_train, y_train)
print("GridSearch SVM finalizado.")
print("Mejor SVM:", svm_grid.best_params_, "accuracy=", svm_grid.best_score_)

# Definir grilla de parámetros para Random Forest
param_grid_rf = {
    "clf__n_estimators": [100, 200],  # Número de árboles
    "clf__max_depth": [None, 5]  # Profundidad máxima
}

print("\nIniciando GridSearch para RandomForest...")
rf_grid = GridSearchCV(
    models["RandomForest"],
    param_grid_rf,
    cv=cv,
    scoring="accuracy",
    n_jobs=1,
    verbose=2
)
rf_grid.fit(X_train, y_train)
print("GridSearch RandomForest finalizado.")
print("Mejor RF:", rf_grid.best_params_, "accuracy=", rf_grid.best_score_)

# Seleccionar el mejor modelo entre SVM y Random Forest optimizados
best_estimator = svm_grid if svm_grid.best_score_ >= rf_grid.best_score_ else rf_grid
best_model = best_estimator.best_estimator_
best_name = "SVM_RBF" if best_estimator is svm_grid else "RandomForest"
print(f"\nModelo seleccionado: {best_name}")

# ==========================================
# 7) Evaluación en test
# ==========================================
print("\n[7/10] Evaluación en conjunto de prueba...")

# Evaluar el mejor modelo en el conjunto de prueba (datos no vistos)
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print("\n=== Resultados en test ===")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=class_names))

# Crear matriz de confusión
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Matriz de confusión - {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# Calcular ROC-AUC si el modelo soporta predict_proba
try:
    y_score = best_model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc = roc_auc_score(y_bin, y_score, multi_class='ovr', average='macro')
    print(f"ROC-AUC (macro, OvR): {auc:.3f}")
except Exception as e:
    print("No se pudo calcular ROC-AUC:", e)

# ==========================================
# 8) Importancia de características
# ==========================================
print("\n[8/10] Calculando importancia de características...")

feature_importances = None
feature_names = X_feat.columns

# Extraer importancia de características del mejor modelo
clf = best_model.named_steps.get('clf')
if hasattr(clf, 'feature_importances_'):  # Para modelos basados en árboles
    feature_importances = clf.feature_importances_
elif hasattr(clf, 'coef_'):  # Para modelos lineales
    feature_importances = np.mean(np.abs(clf.coef_), axis=0)

if feature_importances is not None:
    imp = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    sns.barplot(x=imp.values, y=imp.index, palette="mako")
    plt.title(f"Importancia de características - {best_name}")
    plt.tight_layout()
    plt.savefig("outputs/feature_importances.png", dpi=150)
    plt.close()

# ==========================================
# 9) Análisis de clustering
# ==========================================
print("\n[9/10] Análisis de clustering con K-means...")

# Aplicar K-means clustering (aprendizaje no supervisado)
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
X_scaled = StandardScaler().fit_transform(X_feat)
clusters = kmeans.fit_predict(X_scaled)

# Comparar visualmente clases reales vs clusters
plt.figure(figsize=(10,5))

# Plot clases reales
plt.subplot(1,2,1)
for i, name in enumerate(class_names):
    idx = y == i
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=name, alpha=0.7)
plt.title("Clases reales en espacio PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend()

# Plot clusters
plt.subplot(1,2,2)
for i in range(3):
    idx = clusters == i
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Cluster {i}', alpha=0.7)
plt.title("Clusters K-means en espacio PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/clustering_comparison.png", dpi=150)
plt.close()

# Calcular métrica de similitud entre clusters y clases
ari = adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index entre clases reales y clusters: {ari:.3f}")
print("Esto mide qué tan bien coinciden los clusters con las clases reales.")

# ==========================================
# 10) Guardar modelo
# ==========================================
print("\n[10/10] Guardando modelo y resultados...")

# Guardar el mejor modelo entrenado
dump(best_model, "outputs/iris_best_model.joblib")

# Crear archivo de resumen con los resultados
with open("outputs/summary.txt", "w", encoding="utf-8") as f:
    f.write("Resultados del análisis Iris\n")
    f.write(f"Modelo: {best_name}\n")
    f.write(f"Accuracy test: {acc:.3f}\n")
    f.write(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}\n")
    f.write(f"Adjusted Rand Index (clustering): {ari:.3f}\n")

print("\nProceso completo finalizado correctamente.")
print("Revisa la carpeta 'outputs' para ver los resultados.")
