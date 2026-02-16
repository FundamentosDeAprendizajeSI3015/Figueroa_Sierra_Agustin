import pandas as pd
import numpy as np

# File paths
input_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/genres.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/artist_genres.csv'

# Columns to process
# We need artist and genre. We also keep numerical features for aggregation.
# Based on inspection: 'artists', 'track_genre', 'danceability', etc.
numeric_cols = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
cols_to_load = ['artists', 'track_genre'] + numeric_cols

print("Loading genres.csv...")
try:
    df = pd.read_csv(input_file, usecols=cols_to_load)
except ValueError as e:
    # Fallback if some columns don't exist (e.g. key)
    print(f"Error loading columns: {e}. Loading entire file to check columns.")
    df = pd.read_csv(input_file, nrows=5)
    print("Available columns:", df.columns.tolist())
    raise e

print(f"Original shape: {df.shape}")

# 1. Handle multiple artists
# The user said they are separated by ';'. AND seemingly 'regions.csv' uses something else.
# But here we focus on processing genres.csv.
print("Exploding artists...")
# Drop rows with null artists
df = df.dropna(subset=['artists'])

# Split and explode
df['artist'] = df['artists'].str.split(';')
df_exploded = df.explode('artist')

# Clean artist names (strip whitespace)
df_exploded['artist'] = df_exploded['artist'].str.strip()

print(f"Shape after explosion: {df_exploded.shape}")

# 2. Group by artist
print("Grouping by artist...")

# Define aggregation dictionary
agg_dict = {
    'track_genre': lambda x: list(set(x)), # List of unique genres
}

# Add mean for numeric columns
for col in numeric_cols:
    agg_dict[col] = 'mean'

# Group
df_grouped = df_exploded.groupby('artist').agg(agg_dict).reset_index()

# Rename columns for clarity if needed, but keeping them as is is fine.
# track_genre -> genres (list)

print(f"Unique artists found: {df_grouped.shape[0]}")
print("Example row:")
print(df_grouped.iloc[0])

# Save
print(f"Saving to {output_file}...")
df_grouped.to_csv(output_file, index=False)
print("Done.")
