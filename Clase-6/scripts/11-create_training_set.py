import pandas as pd
import os

# File paths
input_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset_v4.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/train_dataset.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Ensure 10-join_latitude.py completed successfully.")
    exit(1)

print("Loading final_dataset_v4.csv...")
# Load in chunks to filter
chunk_size = 500000
first_chunk = True
total_rows = 0
filtered_rows = 0

try:
    chunks = pd.read_csv(input_file, chunksize=chunk_size)
    
    for chunk in chunks:
        # Filter 1: Genres must not be null (implicit in join logic but safe to keep)
        # Filter 2: Temperature must not be null
        # Filter 3: Latitude/Unemployment must not be null (new)
        
        # Combined filter: check critical columns
        # unemployment_rate comes from the latitude join, so if it's there, likely others are too.
        chunk_clean = chunk.dropna(subset=['track_genre', 'avg_temp', 'unemployment_rate'])
        
        if not chunk_clean.empty:
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk_clean.to_csv(output_file, index=False, mode=mode, header=header)
            first_chunk = False
            filtered_rows += len(chunk_clean)
            
        total_rows += len(chunk)
        print(".", end="", flush=True)
        
    print(f"\nProcessing complete.")
    print(f"Original rows: {total_rows}")
    print(f"Training set rows: {filtered_rows}")
    print(f"Retention rate: {filtered_rows/total_rows*100:.2f}%")
    
    # Validation step: Load the training set to check country distribution
    print("\nValidating training set distribution...")
    if os.path.exists(output_file):
        df_train = pd.read_csv(output_file, usecols=['region'])
        country_counts = df_train['region'].value_counts()
        print(f"Number of countries in training set: {len(country_counts)}")
        print("Top 10 countries by row count:")
        print(country_counts.head(10))
        print("Bottom 5 countries by row count:")
        print(country_counts.tail(5))
    else:
        print("Error: train_dataset.csv was not created (empty result?).")

except Exception as e:
    print(f"Error creating training set: {e}")
