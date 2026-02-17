import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score, 
                             f1_score, confusion_matrix, ConfusionMatrixDisplay, 
                             roc_curve, roc_auc_score)
from scipy.stats import uniform

# =============================================================================
# CONFIGURACIÓN Y CARGA DE DATOS
# =============================================================================

# Configuración de visualización para gráficos elegantes
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_clean_data(filepath):
    """
    Carga el dataset del Titanic y realiza una limpieza inicial proactiva.
    """
    df = pd.read_csv(filepath)
    
    # Imputación de valores faltantes basada en estadísticas descriptivas
    # Usamos la mediana para edad por ser robusta a outliers
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Usamos la moda para el puerto de embarque
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Eliminación de columnas no informativas para el modelado predictivo
    cols_to_drop = ['Cabin', 'Name', 'Ticket', 'PassengerId']
    df.drop(columns=cols_to_drop, inplace=True)
    
    return df

# Cargar los datos
dataset_path = 'Titanic-Dataset.csv'
df = load_and_clean_data(dataset_path)

# =============================================================================
# 1.0.1 REGRESIÓN LINEAL (Predicción de Edad)
# =============================================================================

# Definir características (X) y objetivo (y)
X_lin = df.drop(columns=['Age'])
y_lin = df['Age']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)

# Definir preprocesamiento: Escalado para numéricas y One-Hot para categóricas
numeric_features = ['Fare', 'SibSp', 'Parch', 'Pclass', 'Survived']
categorical_features = ['Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Definición de Pipelines para Ridge y Lasso
# El uso de Pipelines evita el sesgo de datos (data leakage) al escalar
ridge_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge())])
lasso_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', Lasso())])

# Búsqueda aleatoria de hiperparámetros (alpha) con Validación Cruzada
param_dist = {'regressor__alpha': uniform(0, 10)}

print("\n--- Entrenando Regresión Lineal (Ridge & Lasso) ---")
ridge_search = RandomizedSearchCV(ridge_pipeline, param_dist, n_iter=50, cv=5, 
                                  scoring='neg_mean_absolute_error', random_state=42)
lasso_search = RandomizedSearchCV(lasso_pipeline, param_dist, n_iter=50, cv=5, 
                                  scoring='neg_mean_absolute_error', random_state=42)

ridge_search.fit(X_train_lin, y_train_lin)
lasso_search.fit(X_train_lin, y_train_lin)

# Evaluación y Visualización
def evaluate_regression(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name} -> R^2: {r2:.4f}, MAE: {mae:.4f}")
    return y_pred

pred_ridge = evaluate_regression(ridge_search, X_test_lin, y_test_lin, "Ridge")
pred_lasso = evaluate_regression(lasso_search, X_test_lin, y_test_lin, "Lasso")

# =============================================================================
# 1.0.2 REGRESIÓN LOGÍSTICA (Clasificación de Supervivencia)
# =============================================================================

X_log = df.drop(columns=['Survived'])
y_log = df['Survived']

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

# Preprocesador específico para clasificación
num_log = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
cat_log = ['Sex', 'Embarked']
preprocessor_log = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_log),
        ('cat', OneHotEncoder(), cat_log)
    ])

# Pipeline de Regresión Logística
log_pipeline = Pipeline([('preprocessor', preprocessor_log), ('classifier', LogisticRegression())])

print("\n--- Entrenando Regresión Logística ---")
param_dist_log = {'classifier__C': uniform(0.1, 10)}
log_search = RandomizedSearchCV(log_pipeline, param_dist_log, n_iter=30, cv=5, 
                                scoring='accuracy', random_state=42)
log_search.fit(X_train_log, y_train_log)

# Evaluación de Clasificación
y_pred_log = log_search.predict(X_test_log)
print(f"Accuracy: {accuracy_score(y_test_log, y_pred_log):.4f}")
print(f"F1-score: {f1_score(y_test_log, y_pred_log):.4f}")

# =============================================================================
# FUNCIONALIDADES EXTRA PROFESIONALES
# =============================================================================

# 1. Análisis de Importancia de Características
def plot_feature_importance(model, preprocessor, features_num, features_cat, title):
    ohe = preprocessor.named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(features_cat)
    all_features = np.concatenate([features_num, cat_names])
    
    coefs = model.best_estimator_.named_steps['classifier'].coef_[0]
    importance_df = pd.DataFrame({'Feature': all_features, 'Coefficient': coefs})
    importance_df = importance_df.reindex(importance_df.Coefficient.abs().sort_values(ascending=False).index)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Coefficient', y='Feature', palette='viridis')
    plt.title(title)
    plt.axvline(0, color='black', linestyle='--')
    plt.show()

print("\nGenerando gráficas de diagnóstico...")
plot_feature_importance(log_search, preprocessor_log, num_log, cat_log, 
                        "Influencia de Variables en la Supervivencia")

# 2. Curva ROC y AUC Score
y_probs = log_search.predict_proba(X_test_log)[:, 1]
fpr, tpr, _ = roc_curve(y_test_log, y_probs)
auc = roc_auc_score(y_test_log, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Evaluación ROC - Clasificador Titanic')
plt.legend(loc="lower right")
plt.show()

print("\nAnálisis completo. El script ha finalizado exitosamente.")
