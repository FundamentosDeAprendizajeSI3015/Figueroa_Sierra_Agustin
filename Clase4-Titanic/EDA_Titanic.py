# %% [markdown]
# # 🚢 Análisis Exploratorio de Datos (EDA) - Dataset Titanic
# **SI3015 - Fundamentos de Aprendizaje Automático**
#
# Alumno: Agustín Figueroa Sierra
#
# Fecha: 10 de febrero de 2026

# %% [markdown]
# ## 1. Importación de librerías y carga de datos

# %%
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import category_encoders as ce
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

# %%
df = pd.read_csv("Titanic-Dataset.csv")
print("Dimensiones del dataset:", df.shape)
df.head(10)

# %%
df.info()

# %%
df.describe()

# %%
# Valores nulos por columna
nulos = df.isnull().sum()
porcentaje_nulos = (nulos / len(df) * 100).round(2)
pd.DataFrame({"Nulos": nulos, "Porcentaje (%)": porcentaje_nulos}).query("Nulos > 0")

# %% [markdown]
# ### Observaciones iniciales
# - **Age** tiene ~19.87% de valores nulos (177 registros).
# - **Cabin** tiene ~77.10% de valores nulos → se considerará eliminar o transformar.
# - **Embarked** tiene solo 2 valores nulos → se pueden imputar fácilmente.
# - 891 registros, 12 columnas.

# %% [markdown]
# ## 2. Tratamiento de valores nulos

# %%
# Imputar Age con la mediana (robusto ante outliers)
df["Age"].fillna(df["Age"].median(), inplace=True)

# Imputar Embarked con la moda
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Cabin: crear una columna binaria indicando si tiene cabina asignada
df["HasCabin"] = df["Cabin"].notna().astype(int)

# Eliminar columnas no útiles para el análisis
df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"], inplace=True)

print("Valores nulos restantes:")
print(df.isnull().sum())
df.head()

# %% [markdown]
# ## 3. Medidas de Tendencia Central

# %%
columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

tendencia_central = pd.DataFrame({
    "Media": df[columnas_numericas].mean(),
    "Mediana": df[columnas_numericas].median(),
    "Moda": df[columnas_numericas].mode().iloc[0]
}).round(4)

print("=" * 60)
print("MEDIDAS DE TENDENCIA CENTRAL")
print("=" * 60)
tendencia_central

# %% [markdown]
# ### Interpretación - Tendencia Central
# - **Survived**: Media ≈ 0.38 → solo el 38% sobrevivió.
# - **Pclass**: Moda = 3 → la mayoría viajaba en 3ra clase.
# - **Age**: Media ≈ 29.4, Mediana = 28 → distribución ligeramente sesgada a la derecha.
# - **Fare**: Media ≈ 32.2, Mediana ≈ 14.45 → alta asimetría, hay tarifas muy altas.

# %% [markdown]
# ## 4. Medidas de Dispersión

# %%
dispersion = pd.DataFrame({
    "Desv. Estándar": df[columnas_numericas].std(),
    "Varianza": df[columnas_numericas].var(),
    "Rango": df[columnas_numericas].max() - df[columnas_numericas].min(),
    "Mínimo": df[columnas_numericas].min(),
    "Máximo": df[columnas_numericas].max(),
    "Coef. Variación (%)": ((df[columnas_numericas].std() / df[columnas_numericas].mean()) * 100)
}).round(4)

print("=" * 60)
print("MEDIDAS DE DISPERSIÓN")
print("=" * 60)
dispersion

# %% [markdown]
# ### Interpretación - Dispersión
# - **Fare** tiene el coeficiente de variación más alto (~154%), indicando gran variabilidad.
# - **Age** tiene un coeficiente de variación moderado (~45%).
# - **SibSp** y **Parch** tienen alta variabilidad relativa a su media.

# %% [markdown]
# ## 5. Medidas de Posición y Eliminación de Outliers

# %%
# Cuartiles y rango intercuartílico
posicion = pd.DataFrame({
    "Q1 (25%)": df[columnas_numericas].quantile(0.25),
    "Q2 (50%)": df[columnas_numericas].quantile(0.50),
    "Q3 (75%)": df[columnas_numericas].quantile(0.75),
    "IQR": df[columnas_numericas].quantile(0.75) - df[columnas_numericas].quantile(0.25),
    "Percentil 10": df[columnas_numericas].quantile(0.10),
    "Percentil 90": df[columnas_numericas].quantile(0.90)
}).round(4)

print("=" * 60)
print("MEDIDAS DE POSICIÓN")
print("=" * 60)
posicion

# %%
# Boxplots antes de eliminar outliers
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cols_outliers = ["Age", "Fare", "SibSp"]
for i, col in enumerate(cols_outliers):
    sns.boxplot(data=df, y=col, ax=axes[i], color="salmon")
    axes[i].set_title(f"Boxplot de {col} (ANTES de eliminar outliers)", fontsize=12)
plt.tight_layout()
plt.savefig("boxplots_antes.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Función para eliminar outliers usando IQR
def eliminar_outliers_iqr(dataframe, columna, factor=1.5):
    Q1 = dataframe[columna].quantile(0.25)
    Q3 = dataframe[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - factor * IQR
    limite_superior = Q3 + factor * IQR
    n_antes = len(dataframe)
    df_limpio = dataframe[(dataframe[columna] >= limite_inferior) & (dataframe[columna] <= limite_superior)]
    n_despues = len(df_limpio)
    print(f"  {columna}: eliminados {n_antes - n_despues} outliers "
          f"(rango válido: [{limite_inferior:.2f}, {limite_superior:.2f}])")
    return df_limpio

print("Eliminación de outliers con método IQR (factor=1.5):")
print("-" * 50)
df_clean = df.copy()
for col in ["Age", "Fare"]:
    df_clean = eliminar_outliers_iqr(df_clean, col)

print(f"\nRegistros originales: {len(df)} → Registros limpios: {len(df_clean)}")

# %%
# Boxplots después de eliminar outliers
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(cols_outliers):
    sns.boxplot(data=df_clean, y=col, ax=axes[i], color="lightgreen")
    axes[i].set_title(f"Boxplot de {col} (DESPUÉS de eliminar outliers)", fontsize=12)
plt.tight_layout()
plt.savefig("boxplots_despues.png", dpi=150, bbox_inches="tight")
plt.show()

# Seguimos trabajando con df_clean
df = df_clean.copy()

# %% [markdown]
# ## 6. Histogramas - Análisis de Distribución

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
cols_hist = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

for i, (col, color) in enumerate(zip(cols_hist, colors)):
    ax = axes[i // 3, i % 3]
    ax.hist(df[col], bins=30, color=color, edgecolor="black", alpha=0.7)
    ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Media: {df[col].mean():.2f}")
    ax.axvline(df[col].median(), color="blue", linestyle="-.", linewidth=1.5, label=f"Mediana: {df[col].median():.2f}")
    ax.set_title(f"Distribución de {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)

plt.suptitle("Histogramas de Variables Numéricas", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("histogramas.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Distribución de variables categóricas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Survived
survived_counts = df["Survived"].value_counts()
axes[0].bar(["No sobrevivió (0)", "Sobrevivió (1)"], survived_counts.values,
            color=["#e74c3c", "#2ecc71"], edgecolor="black")
axes[0].set_title("Distribución de Supervivencia", fontweight="bold")
axes[0].set_ylabel("Cantidad")
for j, v in enumerate(survived_counts.values):
    axes[0].text(j, v + 5, str(v), ha="center", fontweight="bold")

# Sex
sex_counts = df["Sex"].value_counts()
axes[1].bar(sex_counts.index, sex_counts.values, color=["#3498db", "#e91e63"], edgecolor="black")
axes[1].set_title("Distribución por Sexo", fontweight="bold")
axes[1].set_ylabel("Cantidad")
for j, v in enumerate(sex_counts.values):
    axes[1].text(j, v + 5, str(v), ha="center", fontweight="bold")

# Embarked
embarked_counts = df["Embarked"].value_counts()
axes[2].bar(embarked_counts.index, embarked_counts.values, color=["#f39c12", "#9b59b6", "#1abc9c"], edgecolor="black")
axes[2].set_title("Distribución por Puerto de Embarque", fontweight="bold")
axes[2].set_ylabel("Cantidad")
for j, v in enumerate(embarked_counts.values):
    axes[2].text(j, v + 5, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("distribucion_categoricas.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Interpretación - Distribuciones
# - **Survived**: Más personas no sobrevivieron (~62%) que las que sí (~38%).
# - **Age**: Distribución aproximadamente normal centrada entre 20-35 años.
# - **Fare**: Distribución muy sesgada a la derecha; la mayoría pagó tarifas bajas.
# - **Sex**: Más hombres que mujeres a bordo.
# - **Embarked**: La mayoría embarcó en Southampton (S).

# %% [markdown]
# ## 7. Gráficos de Dispersión - Relación entre Variables

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Age vs Fare
scatter1 = axes[0, 0].scatter(df["Age"], df["Fare"], c=df["Survived"],
                               cmap="RdYlGn", alpha=0.6, edgecolors="gray", s=40)
axes[0, 0].set_xlabel("Edad")
axes[0, 0].set_ylabel("Tarifa")
axes[0, 0].set_title("Edad vs Tarifa (color = Survived)", fontweight="bold")
plt.colorbar(scatter1, ax=axes[0, 0], label="Survived")

# Age vs SibSp
scatter2 = axes[0, 1].scatter(df["Age"], df["SibSp"], c=df["Survived"],
                               cmap="RdYlGn", alpha=0.6, edgecolors="gray", s=40)
axes[0, 1].set_xlabel("Edad")
axes[0, 1].set_ylabel("Hermanos/Cónyuge a bordo")
axes[0, 1].set_title("Edad vs SibSp (color = Survived)", fontweight="bold")
plt.colorbar(scatter2, ax=axes[0, 1], label="Survived")

# Fare vs Pclass
scatter3 = axes[1, 0].scatter(df["Fare"], df["Pclass"], c=df["Survived"],
                               cmap="RdYlGn", alpha=0.6, edgecolors="gray", s=40)
axes[1, 0].set_xlabel("Tarifa")
axes[1, 0].set_ylabel("Clase")
axes[1, 0].set_title("Tarifa vs Clase (color = Survived)", fontweight="bold")
plt.colorbar(scatter3, ax=axes[1, 0], label="Survived")

# Parch vs SibSp
scatter4 = axes[1, 1].scatter(df["Parch"], df["SibSp"], c=df["Survived"],
                               cmap="RdYlGn", alpha=0.6, edgecolors="gray", s=40)
axes[1, 1].set_xlabel("Padres/Hijos a bordo")
axes[1, 1].set_ylabel("Hermanos/Cónyuge a bordo")
axes[1, 1].set_title("Parch vs SibSp (color = Survived)", fontweight="bold")
plt.colorbar(scatter4, ax=axes[1, 1], label="Survived")

plt.suptitle("Gráficos de Dispersión entre Variables", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("scatter_plots.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Supervivencia por clase y sexo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", ax=axes[0], palette="Set2")
axes[0].set_title("Tasa de Supervivencia por Clase y Sexo", fontweight="bold")
axes[0].set_xlabel("Clase")
axes[0].set_ylabel("Tasa de Supervivencia")

sns.violinplot(data=df, x="Survived", y="Age", hue="Sex", split=True, ax=axes[1], palette="Set1")
axes[1].set_title("Distribución de Edad por Supervivencia y Sexo", fontweight="bold")
axes[1].set_xlabel("Sobrevivió")
axes[1].set_ylabel("Edad")

plt.tight_layout()
plt.savefig("supervivencia_clase_sexo.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Interpretación - Dispersión
# - Los pasajeros de **1ra clase** y con **tarifas altas** tuvieron mayor probabilidad de supervivencia.
# - Las **mujeres** tuvieron una tasa de supervivencia significativamente mayor que los hombres en todas las clases.
# - No hay una relación lineal fuerte entre **Edad** y **Tarifa**.
# - Los pasajeros que viajaban **solos** (SibSp=0, Parch=0) tuvieron menor tasa de supervivencia.

# %% [markdown]
# ---
# ## 8. Transformaciones de Columnas

# %% [markdown]
# ### 8.1 One Hot Encoding

# %%
print("Columna 'Embarked' - valores únicos:", df["Embarked"].unique())
print()

# One Hot Encoding para Embarked
df_ohe = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked", drop_first=False, dtype=int)
print("One Hot Encoding aplicado a 'Embarked':")
df_ohe[["Embarked_C", "Embarked_Q", "Embarked_S"]].head(10)

# %% [markdown]
# ### 8.2 Label Encoding

# %%
le = LabelEncoder()
df_ohe["Sex_LabelEnc"] = le.fit_transform(df_ohe["Sex"])
print("Label Encoding aplicado a 'Sex':")
print(f"  Mapeo: {dict(zip(le.classes_, le.transform(le.classes_)))}")
df_ohe[["Sex", "Sex_LabelEnc"]].head(10)

# %% [markdown]
# ### 8.3 Binary Encoding

# %%
# Binary Encoding para Pclass (tiene 3 categorías: 1, 2, 3)
binary_encoder = ce.BinaryEncoder(cols=["Pclass"], return_df=True)
df_binary = binary_encoder.fit_transform(df_ohe[["Pclass"]])
print("Binary Encoding aplicado a 'Pclass':")
df_binary.head(10)

# %%
# Integrar todas las transformaciones
df_transformed = df_ohe.copy()
df_transformed = pd.concat([df_transformed, df_binary.add_prefix("Pclass_bin_")], axis=1)
df_transformed.drop(columns=["Sex"], inplace=True)  # Ya tenemos Sex_LabelEnc

print("Dataset transformado:")
print(f"Dimensiones: {df_transformed.shape}")
df_transformed.head()

# %% [markdown]
# ### 8.4 Correlación entre columnas

# %%
# Calcular la matriz de correlación
columnas_corr = df_transformed.select_dtypes(include=[np.number]).columns
corr_matrix = df_transformed[columnas_corr].corr()

# Heatmap de correlación
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)
plt.title("Matriz de Correlación - Variables Transformadas", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlacion.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Identificar correlaciones altas (> 0.7 o < -0.7) excluyendo la diagonal
print("Correlaciones altas (|r| > 0.7):")
print("=" * 50)
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: "
                  f"{corr_matrix.iloc[i, j]:.4f}")

# %%
# Correlaciones con la variable objetivo (Survived)
corr_survived = corr_matrix["Survived"].drop("Survived").sort_values(ascending=False)
print("\nCorrelación con 'Survived' (ordenada):")
print("-" * 40)
for col, val in corr_survived.items():
    emoji = "🟢" if abs(val) > 0.3 else "🟡" if abs(val) > 0.1 else "⚪"
    print(f"  {emoji} {col}: {val:.4f}")

# %%
# Visualización de correlación con Survived
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_survived.values]
corr_survived.plot(kind="barh", color=colors, edgecolor="black", ax=ax)
ax.set_title("Correlación de cada variable con 'Survived'", fontweight="bold", fontsize=14)
ax.set_xlabel("Coeficiente de Correlación")
ax.axvline(x=0, color="black", linewidth=0.8)
ax.axvline(x=0.3, color="gray", linestyle="--", alpha=0.5, label="|r| = 0.3")
ax.axvline(x=-0.3, color="gray", linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig("correlacion_survived.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Análisis de correlación - ¿Eliminar columnas?
# - Se eliminan columnas con correlación muy baja con `Survived` y/o alta colinealidad entre sí.
# - Las columnas de Binary Encoding de Pclass son redundantes con Pclass original.

# %%
# Eliminar columnas redundantes o de baja relevancia
columnas_a_eliminar = ["Pclass_bin_Pclass_0", "Pclass_bin_Pclass_1"]
if all(c in df_transformed.columns for c in columnas_a_eliminar):
    df_transformed.drop(columns=columnas_a_eliminar, inplace=True)
    print(f"Columnas eliminadas: {columnas_a_eliminar}")

print(f"Columnas restantes ({len(df_transformed.columns)}): {list(df_transformed.columns)}")

# %% [markdown]
# ### 8.5 Escalado de Variables

# %%
columnas_a_escalar = ["Age", "Fare", "SibSp", "Parch"]

# Min-Max Scaling
scaler_mm = MinMaxScaler()
df_minmax = df_transformed.copy()
df_minmax[columnas_a_escalar] = scaler_mm.fit_transform(df_minmax[columnas_a_escalar])

# Standard Scaler
scaler_std = StandardScaler()
df_standard = df_transformed.copy()
df_standard[columnas_a_escalar] = scaler_std.fit_transform(df_standard[columnas_a_escalar])

# Comparación visual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original
df_transformed[columnas_a_escalar].boxplot(ax=axes[0])
axes[0].set_title("Original (sin escalar)", fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)

# Min-Max
df_minmax[columnas_a_escalar].boxplot(ax=axes[1])
axes[1].set_title("Min-Max Scaling [0, 1]", fontweight="bold")
axes[1].tick_params(axis="x", rotation=45)

# Standard Scaler
df_standard[columnas_a_escalar].boxplot(ax=axes[2])
axes[2].set_title("Standard Scaler (μ=0, σ=1)", fontweight="bold")
axes[2].tick_params(axis="x", rotation=45)

plt.suptitle("Comparación de Métodos de Escalado", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("escalado_comparacion.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Estadísticas después de escalar
print("Estadísticas después de Min-Max Scaling:")
print(df_minmax[columnas_a_escalar].describe().round(4))
print("\nEstadísticas después de Standard Scaler:")
print(df_standard[columnas_a_escalar].describe().round(4))

# %% [markdown]
# ### 8.6 Transformación Logarítmica

# %%
# Fare tiene distribución muy sesgada → candidata ideal para log transform
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Antes
axes[0].hist(df_transformed["Fare"], bins=40, color="#e74c3c", edgecolor="black", alpha=0.7)
axes[0].set_title("Fare - Original", fontweight="bold")
axes[0].set_xlabel("Fare")
axes[0].set_ylabel("Frecuencia")
skew_original = df_transformed["Fare"].skew()
axes[0].text(0.7, 0.85, f"Skewness: {skew_original:.4f}", transform=axes[0].transAxes,
             fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

# Transformación logarítmica (log1p para manejar ceros)
df_transformed["Fare_log"] = np.log1p(df_transformed["Fare"])

axes[1].hist(df_transformed["Fare_log"], bins=40, color="#2ecc71", edgecolor="black", alpha=0.7)
axes[1].set_title("Fare - Transformación Log", fontweight="bold")
axes[1].set_xlabel("log(1 + Fare)")
axes[1].set_ylabel("Frecuencia")
skew_log = df_transformed["Fare_log"].skew()
axes[1].text(0.7, 0.85, f"Skewness: {skew_log:.4f}", transform=axes[1].transAxes,
             fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

plt.suptitle("Transformación Logarítmica de Fare", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("log_transform_fare.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSkewness original de Fare:  {skew_original:.4f}")
print(f"Skewness después de log:    {skew_log:.4f}")
print("→ La transformación logarítmica reduce significativamente la asimetría de Fare.")

# %% [markdown]
# ### Verificación de asimetría en otras columnas

# %%
print("Skewness de columnas numéricas:")
print("=" * 40)
for col in columnas_a_escalar:
    skew_val = df_transformed[col].skew()
    necesita = "⚠️ Considerar log transform" if abs(skew_val) > 1 else "✅ OK"
    print(f"  {col}: {skew_val:.4f}  {necesita}")

# %% [markdown]
# ---
# ## 9. Dataset Final Transformado

# %%
print("=" * 60)
print("DATASET FINAL TRANSFORMADO")
print("=" * 60)
print(f"Dimensiones: {df_transformed.shape}")
print(f"\nColumnas: {list(df_transformed.columns)}")
print(f"\nTipos de datos:")
print(df_transformed.dtypes)
print(f"\nPrimeras 5 filas:")
df_transformed.head()

# %%
# Guardar dataset transformado
df_transformed.to_csv("Titanic_Transformed.csv", index=False)
print("Dataset transformado guardado como 'Titanic_Transformed.csv'")

# %% [markdown]
# ---
# ## 10. Conclusiones
#
# ### Hallazgos principales del EDA:
#
# 1. **Supervivencia general**: Solo el ~38% de los pasajeros sobrevivió el naufragio del Titanic,
#    lo que refleja la magnitud de la tragedia.
#
# 2. **Género como factor clave**: Las mujeres tuvieron una tasa de supervivencia significativamente
#    mayor que los hombres (~74% vs ~19%), confirmando la política de "mujeres y niños primero".
#
# 3. **Clase socioeconómica**: Los pasajeros de 1ra clase tuvieron la mayor tasa de supervivencia (~63%),
#    seguidos de 2da clase (~47%) y 3ra clase (~24%). La clase fue un factor determinante.
#
# 4. **Edad**: Los niños pequeños (<10 años) tuvieron mayor probabilidad de sobrevivir.
#    La distribución de edad era aproximadamente normal centrada en ~28 años.
#
# 5. **Tarifa (Fare)**: Variable con distribución altamente sesgada a la derecha. Las tarifas
#    altas correlacionan con mayor supervivencia (probablemente mediada por la clase).
#    La transformación logarítmica mejoró significativamente la distribución.
#
# 6. **Familia a bordo**: Pasajeros con 1-2 familiares tenían mejor tasa de supervivencia
#    que quienes viajaban solos o con familias muy numerosas.
#
# 7. **Puerto de embarque**: La mayoría embarcó en Southampton. Los pasajeros de Cherbourg
#    tuvieron una mayor tasa de supervivencia, probablemente correlacionado con la clase.
#
# 8. **Cabina**: El 77% de los registros no tenían cabina asignada. La presencia de cabina
#    se correlaciona positivamente con la supervivencia (indicador de clase alta).
#
# ### Sobre las transformaciones:
# - **One Hot Encoding** fue aplicado a Embarked (3 categorías), ideal para variables nominales sin orden.
# - **Label Encoding** fue aplicado a Sex (2 categorías), adecuado para variables binarias.
# - **Binary Encoding** fue aplicado a Pclass, proporcionando una representación compacta.
# - La **matriz de correlación** ayudó a identificar columnas redundantes y las variables más
#   relevantes para predecir la supervivencia.
# - **Standard Scaler** y **Min-Max Scaler** normalizan las variables numéricas para algoritmos
#   sensibles a la escala (e.g., SVM, KNN, redes neuronales).
# - La **transformación logarítmica** de Fare redujo la asimetría de 1.37 a valores cercanos a 0,
#   haciendo la distribución más simétrica y apropiada para modelos paramétricos.
#
# ### Recomendaciones para el modelado:
# - Utilizar **Sex**, **Pclass**, **Fare** y **Age** como variables predictoras principales.
# - Considerar **HasCabin** como variable indicadora útil.
# - Aplicar el escalado apropiado según el algoritmo a utilizar.
# - El dataset limpio y transformado está listo para entrenar modelos de clasificación.
