import pandas as pd
import os
import re

# File paths
regions_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/regions.csv'
genres_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/artist_genres.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset.csv'

# Check if input files exist
if not os.path.exists(regions_file):
    raise FileNotFoundError(f"{regions_file} not found.")
if not os.path.exists(genres_file):
    raise FileNotFoundError(f"{genres_file} not found.")

print("Loading genres data...")
df_genres = pd.read_csv(genres_file)
# Create a lookup dictionary: normalized artist name -> original artist name in genres file
# We normalize by lowercasing and stripping whitespace
df_genres['artist'] = df_genres['artist'].apply(str)
genre_artist_map = {name.lower().strip(): name for name in df_genres['artist']}
print(f"Loaded {len(genre_artist_map)} unique artists from genres file.")

print("Loading unique artists from regions file...")
# Read only the 'artist' column to get unique artists efficiently
unique_artists_series = pd.read_csv(regions_file, usecols=['artist'])['artist'].unique()
print(f"Found {len(unique_artists_series)} unique artists in regions file.")

# Smart Matching Logic
print("Building artist mapping dictionary...")
artist_mapping = {}

# Regex for splitting: comma, semicolon, 'feat.', 'ft.', 'with', '&', 'vs.'
split_pattern = re.compile(r'\s*(?:,|;| vs\.? | feat\.? | ft\.? |&| with )\s*', re.IGNORECASE)

matched_count = 0
unmatched_count = 0

for original_artist in unique_artists_series:
    if not isinstance(original_artist, str):
        artist_mapping[original_artist] = None
        unmatched_count += 1
        continue
        
    normalized_artist = original_artist.lower().strip()
    
    # Strategy 1: Exact Match
    if normalized_artist in genre_artist_map:
        artist_mapping[original_artist] = genre_artist_map[normalized_artist]
        matched_count += 1
        continue
        
    # Strategy 2: Smart Split & Match Primary Artist
    parts = split_pattern.split(normalized_artist)
    if parts:
        primary_artist = parts[0].strip()
        if primary_artist in genre_artist_map:
            artist_mapping[original_artist] = genre_artist_map[primary_artist]
            matched_count += 1
            continue
            
    # No match found
    artist_mapping[original_artist] = None
    unmatched_count += 1

print(f"Mapping complete. Matched: {matched_count} ({matched_count/len(unique_artists_series)*100:.2f}%) | Unmatched: {unmatched_count}")

# Convert mapping to DataFrame for merge
mapping_df = pd.DataFrame(list(artist_mapping.items()), columns=['artist', 'join_key'])

print("Processing regions file and joining...")
# Read regions in chunks to manage memory
chunk_size = 500000
first_chunk = True

try:
    chunks = pd.read_csv(regions_file, chunksize=chunk_size)
    
    for i, chunk in enumerate(chunks):
        # Merge chunk with mapping to get join_key
        chunk_mapped = pd.merge(chunk, mapping_df, on='artist', how='left')
        
        # Merge with genres using join_key
        # We drop the 'artist' from genres to avoid duplication, keeping the regions' artist name as primary reference
        # but we use the genre data associated with the matched artist
        chunk_final = pd.merge(chunk_mapped, df_genres, left_on='join_key', right_on='artist', how='left', suffixes=('', '_genre_source'))
        
        # Drop helper columns
        chunk_final = chunk_final.drop(columns=['join_key', 'artist_genre_source'])
        
        # Append to output file
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk_final.to_csv(output_file, index=False, mode=mode, header=header)
        first_chunk = False
        
        print(".", end="", flush=True)

    print(f"\nJoin complete. Saved to {output_file}")

except Exception as e:
    print(f"\nError processing regions file: {e}")
