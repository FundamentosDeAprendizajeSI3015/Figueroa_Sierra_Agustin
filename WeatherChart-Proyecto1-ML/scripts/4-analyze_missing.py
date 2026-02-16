import pandas as pd
import os

# File paths
final_dataset = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset.csv'
genres_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/artist_genres.csv'

# Load unique artists from genres file for comparison
print("Loading unique artists from genres file...")
df_genres = pd.read_csv(genres_file, usecols=['artist'])
genre_artists = set(df_genres['artist'].astype(str).str.lower().str.strip())

print("Analyzing final dataset for missing genres...")
# Process in chunks because file is huge (4GB)
chunk_size = 1000000
missing_artists_counts = {}

try:
    chunks = pd.read_csv(final_dataset, usecols=['artist', 'track_genre'], chunksize=chunk_size)
    
    for i, chunk in enumerate(chunks):
        # Filter rows where genres is null
        missing = chunk[chunk['track_genre'].isna()]
        
        if not missing.empty:
            # Count missing artists in this chunk
            counts = missing['artist'].value_counts()
            for artist, count in counts.items():
                missing_artists_counts[artist] = missing_artists_counts.get(artist, 0) + count
        
        print(".", end="", flush=True)

    print("\nAnalysis complete.")
    
    # Calculate total rows processed (approx)
    total_rows = 26173514 # Hardcoded from previous run knowledge or just omit
    # Actually, we can sum counts if we tracked total missing
    
    # Convert to list and sort
    missing_list = sorted(missing_artists_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Total unique artists with missing genres: {len(missing_list)}")
    print("-" * 30)
    print("TOP 20 MISSING ARTISTS:")
    print(f"{'Artist':<40} | {'Count':<10}")
    print("-" * 55)
    
    for artist, count in missing_list[:20]:
        print(f"{str(artist)[:40]:<40} | {count:<10}")
        
    print("-" * 55)
    
    # Check match attempt
    print("\nChecking top 20 against genre database (case-insensitive):")
    for artist, count in missing_list[:20]:
        clean_artist = str(artist).lower().strip()
        if clean_artist in genre_artists:
            print(f"MATCH FOUND: '{artist}' exists in genres file! (Why wasn't it matched?)")
        else:
            print(f"NO MATCH:    '{artist}'")

except Exception as e:
    print(f"Error analyzing file: {e}")
