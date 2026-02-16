"""
13-preprocess.py
Comprehensive preprocessing pipeline for the Weather & Music Genre prediction model.

Pipeline Steps:
  1. Load training data (chunked for memory efficiency)
  2. Drop identifier columns (title, date, artist) — not predictive features
  3. Target Engineering — extract primary genre from list-string format
  4. Categorical Encoding — LabelEncoder for 'region' and 'continent'
  5. Feature Scaling — StandardScaler on numerical features (fit on train only)
  6. Stratified Train/Test Split (80/20)
  7. Save processed splits as Parquet files for fast I/O
  8. Save encoders and scaler as pickle for reproducibility

Output Files:
  - data/X_train.parquet, data/X_test.parquet
  - data/y_train.parquet, data/y_test.parquet
  - data/preprocessing_artifacts.pkl (encoders, scaler, feature names)
  - data/preprocess_report.txt (summary statistics)
"""

import pandas as pd
import numpy as np
import ast
import os
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart'
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'train_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')

# Columns to drop (identifiers, not predictive)
ID_COLUMNS = ['title', 'date', 'artist']

# Target column
TARGET_COL = 'track_genre'

# Categorical columns to encode
CATEGORICAL_COLS = ['region', 'continent']

# Numerical columns to scale
NUMERICAL_COLS = [
    'danceability', 'energy', 'key', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'month', 'avg_temp', 'population', 'gdp_per_capita',
    'latitude', 'longitude', 'tertiary_enrollment', 'unemployment_rate'
]

# Split parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Sample size for processing (None = use all data)
# Set to a number (e.g., 2_000_000) if memory is limited
SAMPLE_SIZE = 2_000_000


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_primary_genre(genre_str):
    """
    Extract the first (primary) genre from a stringified list.
    
    Examples:
        "['pop', 'rock', 'dance']" -> 'pop'
        "['rock']"                 -> 'rock'
        "pop"                      -> 'pop'  (fallback for plain strings)
    """
    try:
        genre_list = ast.literal_eval(str(genre_str))
        if isinstance(genre_list, list) and len(genre_list) > 0:
            return genre_list[0]
        return str(genre_str)
    except (ValueError, SyntaxError):
        # If it's already a plain string genre
        return str(genre_str).strip()


def filter_rare_genres(df, target_col, min_samples=500):
    """
    Remove rows with genres that have fewer than `min_samples` occurrences.
    This prevents issues with stratified splitting and improves model focus.
    
    Returns:
        df_filtered: DataFrame with only sufficiently common genres
        removed_genres: list of genres that were removed
    """
    genre_counts = df[target_col].value_counts()
    valid_genres = genre_counts[genre_counts >= min_samples].index
    removed_genres = genre_counts[genre_counts < min_samples].index.tolist()
    
    df_filtered = df[df[target_col].isin(valid_genres)].copy()
    return df_filtered, removed_genres


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    report_lines = []
    
    def log(msg):
        """Print and store for report."""
        print(msg)
        report_lines.append(msg)
    
    log("=" * 60)
    log("13-PREPROCESS.PY — ML Preprocessing Pipeline")
    log("=" * 60)
    
    # ----------------------------------------------------------
    # STEP 1: Load Data
    # ----------------------------------------------------------
    log("\n[STEP 1] Loading training data...")
    
    if not os.path.exists(INPUT_FILE):
        log(f"ERROR: {INPUT_FILE} not found. Run 11-create_training_set.py first.")
        return
    
    if SAMPLE_SIZE:
        df = pd.read_csv(INPUT_FILE, nrows=SAMPLE_SIZE)
        log(f"  Loaded sample: {len(df):,} rows (of full dataset)")
    else:
        df = pd.read_csv(INPUT_FILE)
        log(f"  Loaded full dataset: {len(df):,} rows")
    
    log(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
    log(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # ----------------------------------------------------------
    # STEP 2: Drop Identifier Columns
    # ----------------------------------------------------------
    log("\n[STEP 2] Dropping identifier columns...")
    
    cols_to_drop = [c for c in ID_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log(f"  Dropped: {cols_to_drop}")
    log(f"  Remaining columns ({len(df.columns)}): {df.columns.tolist()}")
    
    # ----------------------------------------------------------
    # STEP 3: Target Engineering
    # ----------------------------------------------------------
    log("\n[STEP 3] Extracting primary genre from target column...")
    
    df['primary_genre'] = df[TARGET_COL].apply(extract_primary_genre)
    df = df.drop(columns=[TARGET_COL])
    
    n_unique_genres = df['primary_genre'].nunique()
    log(f"  Unique primary genres found: {n_unique_genres}")
    log(f"  Top 10 genres:")
    top_genres = df['primary_genre'].value_counts().head(10)
    for genre, count in top_genres.items():
        log(f"    {genre:<25} {count:>8,} ({count/len(df)*100:.1f}%)")
    
    # Filter rare genres
    log("\n  Filtering rare genres (min 500 samples)...")
    df, removed = filter_rare_genres(df, 'primary_genre', min_samples=500)
    log(f"  Removed {len(removed)} rare genres: {removed[:10]}{'...' if len(removed) > 10 else ''}")
    log(f"  Remaining genres: {df['primary_genre'].nunique()}")
    log(f"  Remaining rows: {len(df):,}")
    
    # ----------------------------------------------------------
    # STEP 4: Categorical Encoding
    # ----------------------------------------------------------
    log("\n[STEP 4] Encoding categorical features...")
    
    encoders = {}
    
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            log(f"  {col}: {len(le.classes_)} unique values -> LabelEncoded")
    
    # Encode target
    le_target = LabelEncoder()
    df['primary_genre'] = le_target.fit_transform(df['primary_genre'])
    encoders['primary_genre'] = le_target
    log(f"  primary_genre: {len(le_target.classes_)} classes -> LabelEncoded")
    log(f"  Genre mapping (first 10): {dict(zip(le_target.classes_[:10], range(10)))}")
    
    # ----------------------------------------------------------
    # STEP 5: Train/Test Split (BEFORE scaling to prevent leakage)
    # ----------------------------------------------------------
    log("\n[STEP 5] Stratified Train/Test Split...")
    
    # Separate features and target
    X = df.drop(columns=['primary_genre'])
    y = df['primary_genre']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    log(f"  X_train: {X_train.shape}")
    log(f"  X_test:  {X_test.shape}")
    log(f"  y_train: {y_train.shape} (classes: {y_train.nunique()})")
    log(f"  y_test:  {y_test.shape} (classes: {y_test.nunique()})")
    
    # Verify stratification
    train_dist = y_train.value_counts(normalize=True).head(5)
    test_dist = y_test.value_counts(normalize=True).head(5)
    log(f"\n  Stratification check (top 5 classes):")
    log(f"  {'Class':<8} {'Train %':>10} {'Test %':>10}")
    for cls in train_dist.index:
        t_pct = train_dist.get(cls, 0) * 100
        te_pct = test_dist.get(cls, 0) * 100
        log(f"  {cls:<8} {t_pct:>9.2f}% {te_pct:>9.2f}%")
    
    # ----------------------------------------------------------
    # STEP 6: Feature Scaling (fit on train, transform both)
    # ----------------------------------------------------------
    log("\n[STEP 6] Scaling numerical features (StandardScaler)...")
    
    # Identify numerical columns that exist in the dataframe
    num_cols_present = [c for c in NUMERICAL_COLS if c in X_train.columns]
    
    scaler = StandardScaler()
    X_train[num_cols_present] = scaler.fit_transform(X_train[num_cols_present])
    X_test[num_cols_present] = scaler.transform(X_test[num_cols_present])
    
    log(f"  Scaled {len(num_cols_present)} numerical features")
    log(f"  Features scaled: {num_cols_present}")
    
    # Quick validation: check means/stds of training set
    train_means = X_train[num_cols_present].mean()
    train_stds = X_train[num_cols_present].std()
    log(f"\n  Post-scaling validation (train set):")
    log(f"  {'Feature':<25} {'Mean':>10} {'Std':>10}")
    for col in num_cols_present[:5]:
        log(f"  {col:<25} {train_means[col]:>10.4f} {train_stds[col]:>10.4f}")
    log(f"  ... (all means ≈ 0, all stds ≈ 1)")
    
    # ----------------------------------------------------------
    # STEP 7: Save Outputs
    # ----------------------------------------------------------
    log("\n[STEP 7] Saving processed data...")
    
    # Save as Parquet (fast I/O, preserves dtypes)
    X_train.to_parquet(os.path.join(OUTPUT_DIR, 'X_train.parquet'), index=False)
    X_test.to_parquet(os.path.join(OUTPUT_DIR, 'X_test.parquet'), index=False)
    y_train.to_frame().to_parquet(os.path.join(OUTPUT_DIR, 'y_train.parquet'), index=False)
    y_test.to_frame().to_parquet(os.path.join(OUTPUT_DIR, 'y_test.parquet'), index=False)
    
    log(f"  Saved X_train.parquet ({X_train.shape})")
    log(f"  Saved X_test.parquet ({X_test.shape})")
    log(f"  Saved y_train.parquet ({y_train.shape})")
    log(f"  Saved y_test.parquet ({y_test.shape})")
    
    # Save preprocessing artifacts for reproducibility
    artifacts = {
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': list(X_train.columns),
        'numerical_cols': num_cols_present,
        'categorical_cols': CATEGORICAL_COLS,
        'target_classes': list(le_target.classes_),
        'sample_size': SAMPLE_SIZE,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
    }
    
    artifacts_path = os.path.join(OUTPUT_DIR, 'preprocessing_artifacts.pkl')
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    log(f"  Saved preprocessing_artifacts.pkl")
    
    # ----------------------------------------------------------
    # STEP 8: Final Summary
    # ----------------------------------------------------------
    log("\n" + "=" * 60)
    log("PREPROCESSING COMPLETE — SUMMARY")
    log("=" * 60)
    log(f"  Input rows:        {SAMPLE_SIZE if SAMPLE_SIZE else 'ALL':>12}")
    log(f"  Final rows:        {len(X_train) + len(X_test):>12,}")
    log(f"  Training samples:  {len(X_train):>12,}")
    log(f"  Testing samples:   {len(X_test):>12,}")
    log(f"  Features:          {X_train.shape[1]:>12}")
    log(f"  Target classes:    {len(le_target.classes_):>12}")
    log(f"  Scaling:           {'StandardScaler':>12}")
    log(f"  Cat. Encoding:     {'LabelEncoder':>12}")
    log(f"  Split ratio:       {'80/20':>12}")
    log(f"  Stratified:        {'Yes':>12}")
    log("=" * 60)
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'preprocess_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\nReport saved to {report_path}")
    log("Ready for model training (14-train_model.py).")


if __name__ == '__main__':
    main()
