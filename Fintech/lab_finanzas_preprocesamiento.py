
# This file contains preprocessing and initial exploration code for financial data.
# =============================================================
# LABORATORIO DE PREPROCESAMIENTO Y EXPLORACIÓN INICIAL (FINANZAS)
# Dataset personalizado (CSV) — enfoque reproducible y parametrizable
# =============================================================
# Este script cubre:
# 1) Carga robusta de un CSV financiero
# 2) Exploración inicial (EDA básico)
# 3) Limpieza y tipificación (fechas, numéricos, categóricos)
# 4) Tratamiento de faltantes y outliers (IQR / winsorización opcional)
# 5) Ingeniería ligera de rasgos (retornos y log-retornos)
# 6) Escalado sin fuga (ajustar con train, aplicar a test)
# 7) Partición (aleatoria o por tiempo)
# 8) Exportación a Parquet + diccionario de datos (JSON)
# =============================================================
# USO BÁSICO:
#   python lab_finanzas_preprocesamiento.py #       --input ./data_input/finance_data.csv #       --date-col Fecha #       --id-cols Ticker #       --price-cols Close #       --categorical-cols Sector,Country #       --target-col Target (opcional) #       --time-split (si hay columna de fecha) #       --split-date 2022-01-01 (opcional para time-split) #       --test-size 0.2 #       --winsorize 0.01 0.99 (opcional)
# 
# NOTAS:
# - No genera datos sintéticos. Requiere un CSV provisto por el docente.
# - Si no se especifican columnas, el script intentará inferir tipos.
# =============================================================

# Importación de librerías necesarias
import argparse  # Para parsear argumentos de línea de comandos
from pathlib import Path  # Para manejar rutas de archivos
import json  # Para trabajar con archivos JSON
import warnings  # Para controlar advertencias
warnings.filterwarnings("ignore")  # Ignorar advertencias
import numpy as np  # Para operaciones numéricas
import pandas as pd  # Para manipulación de datos
from sklearn.model_selection import train_test_split  # Para dividir datos
from sklearn.preprocessing import StandardScaler  # Para escalado

# -------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------

def try_read_csv(path, sep, encoding):
    """Intenta leer CSV con utf-8 y fallback a latin-1 si falla."""
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except UnicodeDecodeError:
        print("[WARN] Problema de encoding. Reintentando con 'latin-1'.")
        return pd.read_csv(path, sep=sep, encoding="latin-1")


def coerce_numeric(df, cols=None):
    """Convierte columnas a numérico con coerce."""
    if cols is None:
        cols = df.select_dtypes(include=["object", "string"]).columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def winsorize_df(df, numeric_cols, lower_q=0.01, upper_q=0.99):
    for c in numeric_cols:
        lo = df[c].quantile(lower_q)
        hi = df[c].quantile(upper_q)
        df[c] = df[c].clip(lo, hi)
    return df


def iqr_outlier_mask(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def parse_list_arg(arg):
    if arg is None or len(arg.strip()) == 0:
        return []
    # admite lista separada por coma
    return [a.strip() for a in arg.split(",") if a.strip()]


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Laboratorio de preprocesamiento para dataset financiero personalizado")
parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
parser.add_argument("--sep", default=",", help="Separador de campos (por defecto ',')")
parser.add_argument("--encoding", default="utf-8", help="Encoding del archivo (por defecto 'utf-8')")

parser.add_argument("--date-col", default=None, help="Nombre de la columna de fecha")
parser.add_argument("--id-cols", default="", help="Columnas identificadoras, separadas por coma")
parser.add_argument("--categorical-cols", default="", help="Columnas categóricas, separadas por coma")
parser.add_argument("--numeric-cols", default="", help="Columnas numéricas, separadas por coma (si no, se infieren)")
parser.add_argument("--price-cols", default="", help="Columnas de precio para calcular retornos (coma)")
parser.add_argument("--target-col", default=None, help="Columna objetivo (opcional)")

parser.add_argument("--missing-tokens", default="NA,N/A,na,NaN,?,-999,", help="Tokens a tratar como missing (coma)")
parser.add_argument("--time-split", action="store_true", help="Si se usa, particiona por tiempo usando --date-col")
parser.add_argument("--split-date", default=None, help="Fecha de corte para time-split (YYYY-MM-DD)")
parser.add_argument("--test-size", type=float, default=0.2, help="Proporción de test (random split)")

parser.add_argument("--winsorize", nargs=2, type=float, default=None, help="Cuantiles inferior y superior para winsorización, ej. 0.01 0.99")

parser.add_argument("--outdir", default="./data_output_finanzas", help="Directorio de salida")

args = parser.parse_args()

# -------------------------------------------------------------
# 1) CARGA
# -------------------------------------------------------------
print_section("1) CARGA DEL CSV")
input_path = Path(args.input)
if not input_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

missing_values = parse_list_arg(args.missing_tokens)
df = try_read_csv(input_path, sep=args.sep, encoding=args.encoding)

# Normalización básica de strings (espacios, etc.)
obj_cols = df.select_dtypes(include=["object", "string"]).columns
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip()
    # reemplazo de tokens de missing
    df[c] = df[c].replace({tok: np.nan for tok in missing_values})

print("Shape:", df.shape)
print("Columnas:", list(df.columns))

# -------------------------------------------------------------
# 2) TIPIFICACIÓN Y FECHAS
# -------------------------------------------------------------
print_section("2) TIPIFICACIÓN Y FECHAS")
if args.date_col and args.date_col in df.columns:
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    print(f"Fechas parseadas en '{args.date_col}'. Nulos en fecha:", df[args.date_col].isna().sum())
else:
    print("[INFO] No se especificó --date-col o no existe en el CSV.")

id_cols = parse_list_arg(args.id_cols)
cat_cols_user = parse_list_arg(args.categorical_cols)
num_cols_user = parse_list_arg(args.numeric_cols)
price_cols = parse_list_arg(args.price_cols)

# Inferencia de tipos básica si el usuario no provee listas
if not num_cols_user:
    # detecta numéricos nativos
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
else:
    num_cols = [c for c in num_cols_user if c in df.columns]

if not cat_cols_user:
    # categóricas objeto/categoría excluyendo id/date/target
    exclude = set(id_cols + ([args.date_col] if args.date_col else []) + ([args.target_col] if args.target_col else []))
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]
else:
    cat_cols = [c for c in cat_cols_user if c in df.columns]

print("Numéricas:", num_cols)
print("Categóricas:", cat_cols)

# -------------------------------------------------------------
# 3) EXPLORACIÓN INICIAL
# -------------------------------------------------------------
print_section("3) EXPLORACIÓN INICIAL (EDA)")
print(df.info())
print("\nDescribe (numéricos):")
print(df[num_cols].describe().T)
print("\nValores nulos por columna:")
print(df.isna().sum().sort_values(ascending=False).head(20))
print("\nValores únicos (top 10):")
print(df.nunique().sort_values(ascending=False).head(10))

# -------------------------------------------------------------
# 4) LIMPIEZA Y OUTLIERS
# -------------------------------------------------------------
print_section("4) LIMPIEZA Y OUTLIERS")
# Convertir strings numéricos a numeric cuando sea posible
for c in df.columns:
    if c not in ([args.date_col] if args.date_col else []):
        # intento ligero de conversión sin afectar texto general
        df[c] = pd.to_numeric(df[c], errors="ignore")

# Imputación simple para numéricos y categóricos
for c in num_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna("__MISSING__")

# Outliers por IQR (reporte)
outlier_report = {}
for c in num_cols:
    mask = iqr_outlier_mask(df[c].astype(float)) if pd.api.types.is_numeric_dtype(df[c]) else pd.Series(False, index=df.index)
    outlier_report[c] = int(mask.sum())
print("Posibles outliers detectados (IQR, conteo por columna):")
print(outlier_report)

# Winsorización opcional
if args.winsorize is not None:
    lq, uq = args.winsorize
    print(f"Aplicando winsorización a numéricos en cuantiles [{lq}, {uq}] ...")
    df[num_cols] = winsorize_df(df[num_cols].copy(), num_cols, lower_q=lq, upper_q=uq)

# -------------------------------------------------------------
# 5) INGENIERÍA DE RASGOS (RETORNOS)
# -------------------------------------------------------------
print_section("5) INGENIERÍA DE RASGOS (RETORNOS)")
return_cols = []
logret_cols = []

if price_cols:
    # si hay fecha, ordenar por fecha + ids si existen
    if args.date_col and args.date_col in df.columns:
        sort_cols = ([args.date_col] if args.date_col else [])
        if id_cols:
            sort_cols = id_cols + sort_cols
        df = df.sort_values(sort_cols)
    for pc in price_cols:
        if pc in df.columns and pd.api.types.is_numeric_dtype(df[pc]):
            # retornos por grupo si existen ids, de lo contrario global
            if id_cols:
                df[pc+"_ret"] = df.groupby(id_cols)[pc].pct_change()
                df[pc+"_logret"] = np.log1p(df[pc+"_ret"])  # log(1+r)
            else:
                df[pc+"_ret"] = df[pc].pct_change()
                df[pc+"_logret"] = np.log1p(df[pc+"_ret"])  
            return_cols.append(pc+"_ret")
            logret_cols.append(pc+"_logret")
        else:
            print(f"[WARN] price-col '{pc}' no numérica o no encontrada; se omite.")
else:
    print("[INFO] No se especificaron --price-cols; se omite cálculo de retornos.")

# Imputación simple de retornos iniciales generados (que producen NA en el primer valor de cada grupo)
for c in return_cols + logret_cols:
    if c in df.columns and df[c].isna().any():
        df[c] = df[c].fillna(0.0)

# Actualizar listas de numéricos tras ingeniería
num_cols = sorted(list(set(num_cols + return_cols + logret_cols)))

# -------------------------------------------------------------
# 6) SEPARACIÓN X/y, ESCALADO SIN FUGA
# -------------------------------------------------------------
print_section("6) SEPARACIÓN X/y Y ESCALADO")
# Definir target si existe
y = None
if args.target_col and args.target_col in df.columns:
    y = df[args.target_col].copy()
    X = df.drop(columns=[args.target_col])
else:
    X = df.copy()

# Quitar columnas no predictoras obvias (ids, fecha)
for dropcol in id_cols:
    if dropcol in X.columns:
        X = X.drop(columns=[dropcol])
if args.date_col and args.date_col in X.columns:
    X = X.drop(columns=[args.date_col])

# One-hot para categóricas restantes
cat_in_X = [c for c in cat_cols if c in X.columns]
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)

# Escalado solo de numéricos
num_in_X = [c for c in num_cols if c in X.columns]
scaler = StandardScaler()

# Partición
if args.time_split and args.date_col and args.date_col in df.columns:
    print("Partición por tiempo activada.")
    if not args.split_date:
        raise ValueError("--time-split requiere --split-date (YYYY-MM-DD)")
    cutoff = pd.to_datetime(args.split_date)
    idx_train = df[args.date_col] < cutoff
    idx_test  = df[args.date_col] >= cutoff
    X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()
    y_train = y.loc[idx_train].copy() if y is not None else None
    y_test  = y.loc[idx_test].copy() if y is not None else None
else:
    print("Partición aleatoria estratificada (si y es categórica).")
    if y is not None and (y.dtype == 'object' or pd.api.types.is_integer_dtype(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test = train_test_split(
            X, test_size=args.test_size, random_state=42
        )
        y_train = y.loc[X_train.index] if y is not None else None
        y_test  = y.loc[X_test.index] if y is not None else None

# Ajustar escalador en TRAIN y transformar ambos
if num_in_X:
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_test[num_in_X]  = scaler.transform(X_test[num_in_X])
else:
    print("[INFO] No hay columnas numéricas para escalar tras las transformaciones.")

print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)

# -------------------------------------------------------------
# 7) EXPORTACIÓN Y DICCIONARIO DE DATOS
# -------------------------------------------------------------
print_section("7) EXPORTACIÓN")
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# Reconstruir conjuntos con target si existe
def concat_target(Xdf, yser):
    if yser is None:
        return Xdf
    return Xdf.assign(**{args.target_col: yser})

train_df = concat_target(X_train, y_train)
test_df  = concat_target(X_test, y_test)

train_path = outdir / "finance_train.parquet"
test_path  = outdir / "finance_test.parquet"
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)

print("Archivos exportados:")
print(" -", train_path)
print(" -", test_path)

# Diccionario de datos básico
schema = {
    "source_file": str(input_path),
    "date_col": args.date_col,
    "id_cols": id_cols,
    "categorical_cols": cat_cols,
    "numeric_cols": num_cols,
    "price_cols": price_cols,
    "target_col": args.target_col,
    "time_split": args.time_split,
    "split_date": args.split_date,
    "test_size": args.test_size,
    "winsorize": args.winsorize,
    "X_train_shape": X_train.shape,
    "X_test_shape": X_test.shape,
}

with open(outdir / "finance_data_dictionary.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, ensure_ascii=False, indent=2, default=str)

print("Diccionario de datos guardado en:", outdir / "finance_data_dictionary.json")
print("\nLaboratorio de finanzas finalizado correctamente.")
