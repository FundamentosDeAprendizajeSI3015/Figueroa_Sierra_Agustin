import pandas as pd
import os

# Define file paths
input_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/charts.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/charts_cleaned.csv'

# Define columns to keep
columns_to_keep = ['title', 'date', 'artist', 'region']

# Process in chunks to handle large file size
chunk_size = 100000
chunks = pd.read_csv(input_file, usecols=columns_to_keep, chunksize=chunk_size)

try:
    first_chunk = True
    for chunk in chunks:
        # Write to CSV
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(output_file, index=False, mode=mode, header=header)
        first_chunk = False
        print(".", end="", flush=True) # Progress indicator
    print("\nFiltering complete. Saved to", output_file)
except Exception as e:
    print(f"\nError processing file: {e}")
