"""
8-join_economy.py
Joins the main dataset (final_dataset_v2.csv) with socioeconomic data
(country_economy.csv) to produce final_dataset_v3.csv.

Join strategy:
  - Normalize country names (lowercase, strip) for matching.
  - Left join on region == country.
  - Report matched/unmatched regions.
"""
import pandas as pd

# File paths
main_dataset_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset_v2.csv'
economy_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/country_economy.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset_v3.csv'

# --- Load economy data ---
print("Loading economy data...")
df_economy = pd.read_csv(economy_file)
print(f"Economy data shape: {df_economy.shape}")

# Normalization helper
def normalize_name(name):
    return str(name).lower().strip()

# Build normalized lookup: norm_name -> original_name
economy_country_map = {normalize_name(c): c for c in df_economy['country'].unique()}
print(f"Unique economy countries: {len(economy_country_map)}")

# --- Discover regions in main dataset ---
print("Loading unique regions from main dataset...")
unique_regions = pd.read_csv(main_dataset_file, usecols=['region'])['region'].unique()
print(f"Unique regions: {len(unique_regions)}")

# --- Build region -> economy_country mapping ---
region_map = {}
matched = 0
unmatched_list = []

for region in unique_regions:
    norm = normalize_name(region)
    if norm in economy_country_map:
        region_map[region] = economy_country_map[norm]
        matched += 1
    else:
        region_map[region] = None
        unmatched_list.append(region)

print(f"Matched regions: {matched}/{len(unique_regions)}")
if unmatched_list:
    print(f"Unmatched regions: {unmatched_list}")

# --- Prepare mapping dataframe ---
mapping_df = pd.DataFrame(list(region_map.items()), columns=['region', 'join_country'])

# Rename economy country column for join
df_economy = df_economy.rename(columns={'country': 'join_country'})

# --- Process in chunks ---
print("Joining economy data...")
chunk_size = 500000
first_chunk = True

try:
    chunks = pd.read_csv(main_dataset_file, chunksize=chunk_size)
    for chunk in chunks:
        # Add join_country via mapping
        chunk = pd.merge(chunk, mapping_df, on='region', how='left')
        
        # Join with economy data
        chunk_merged = pd.merge(chunk, df_economy, on='join_country', how='left')
        
        # Drop join helper column
        chunk_merged = chunk_merged.drop(columns=['join_country'])
        
        # Save
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk_merged.to_csv(output_file, index=False, mode=mode, header=header)
        first_chunk = False
        print(".", end="", flush=True)

    print(f"\nJoin complete. Saved to {output_file}")

except Exception as e:
    print(f"Error joining economy data: {e}")
