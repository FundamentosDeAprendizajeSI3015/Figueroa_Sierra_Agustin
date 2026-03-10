"""
12-eda.py
Performs Exploratory Data Analysis (EDA) on the training dataset.
Generates professional statistical summaries and high-quality visualizations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Settings
input_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/train_dataset.csv'
output_dir = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/plots/'
report_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/eda_report_professional.txt'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set professional aesthetic
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
palette = sns.color_palette("viridis")

# 1. Load Data (Sample if large)
print("Loading data sample (1M rows)...")
try:
    # Read a sample for visualization to manage memory
    df = pd.read_csv(input_file, nrows=1000000)
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

import io

# 2. Statistical Summary
print("Generating comprehensive statistical summary...")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("Professional EDA Report\n")
    f.write("=======================\n\n")
    f.write(f"Shape of sample: {df.shape}\n\n")
    
    f.write("--- Data Types & Missing Values ---\n")
    buffer = io.StringIO()
    df.info(buf=buffer)
    f.write(buffer.getvalue() + "\n\n")
    
    f.write(f"Total Missing Values:\n{df.isnull().sum()}\n\n")
    
    f.write("--- Numerical Statistics ---\n")
    f.write(str(df.describe().T) + "\n\n")
    
    f.write("--- Categorical Statistics ---\n")
    f.write(str(df.describe(include=['O']).T) + "\n\n")
    
    f.write("--- Top 20 Countries (by Data Volume) ---\n")
    f.write(str(df['region'].value_counts().head(20)) + "\n\n")
    
    f.write("--- Top 20 Genres ---\n")
    f.write(str(df['track_genre'].value_counts().head(20)) + "\n\n")

print(f"Report saved to {report_file}")

# 3. Visualizations
print("Generating professional plots...")

# Feature Groups
audio_feats = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
               'instrumentalness', 'liveness', 'valence', 'tempo']
socio_feats = ['avg_temp', 'population', 'gdp_per_capita', 'tertiary_enrollment', 
               'unemployment_rate']
geo_feats = ['latitude', 'longitude']

# A. Advanced Histograms & KDE for Audio Features
print("Plotting distributions...")
plt.figure(figsize=(20, 15))
for i, col in enumerate(audio_feats):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30, color=palette[i % len(palette)], edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}', fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f"{output_dir}/dist_audio_features.png", dpi=300)
plt.close()

# B. Boxplots for Outlier Detection (Normalized for view)
print("Plotting boxplots...")
plt.figure(figsize=(16, 8))
# Normalize audio features for side-by-side comparison (min-max scaling just for plot)
df_norm = (df[audio_feats] - df[audio_feats].min()) / (df[audio_feats].max() - df[audio_feats].min())
sns.boxplot(data=df_norm, palette="viridis")
plt.title('Audio Features Boxplots (Normalized)', fontweight='bold')
plt.xticks(rotation=45)
plt.savefig(f"{output_dir}/boxplot_audio_features.png", dpi=300)
plt.close()

# C. Correlation Heatmap (Enhanced)
print("Plotting correlation matrix...")
plt.figure(figsize=(16, 14))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, square=True, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=16)
plt.savefig(f"{output_dir}/correlation_matrix_professional.png", dpi=300)
plt.close()

# D. Geospatial Distribution (Colored by Temp)
if 'latitude' in df.columns and 'longitude' in df.columns:
    print("Plotting geospatial map...")
    plt.figure(figsize=(16, 10))
    sc = plt.scatter(x=df['longitude'], y=df['latitude'], c=df['avg_temp'], 
                     cmap='coolwarm', alpha=0.5, s=10, edgecolor='none')
    plt.colorbar(sc, label='Average Temperature (°C)')
    plt.title('Global Data Distribution (Colored by Avg Temp)', fontweight='bold', fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{output_dir}/geo_scatter_temp.png", dpi=300)
    plt.close()

# E. Top Genres Horizontal Bar Chart
if 'track_genre' in df.columns:
    print("Plotting top genres...")
    plt.figure(figsize=(12, 10))
    top_genres = df['track_genre'].value_counts().iloc[:20]
    sns.barplot(x=top_genres.values, y=top_genres.index, palette="mako")
    plt.title('Top 20 Genres Frequency', fontweight='bold', fontsize=14)
    plt.xlabel('Number of Songs')
    plt.ylabel('Genre/Label')
    plt.savefig(f"{output_dir}/top_genres_bar.png", dpi=300)
    plt.close()

# F. Audio Features Violin Plot by Continent (if available)
if 'continent' in df.columns:
    print("Plotting violin plots by continent...")
    plt.figure(figsize=(14, 8))
    # Let's plot 'energy' distribution by continent
    sns.violinplot(x='continent', y='energy', data=df, palette="Set2", split=True)
    plt.title('Energy Distribution by Continent', fontweight='bold')
    plt.savefig(f"{output_dir}/violin_energy_continent.png", dpi=300)
    plt.close()

print("Professional EDA Complete.")
