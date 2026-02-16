"""
10-join_latitude.py
Joins the main dataset (final_dataset_v3.csv) with geographic/socioeconomic data
(country_latitude.csv) to produce final_dataset_v4.csv.

Join strategy:
  - Normalize country names (lowercase, strip) for matching.
  - Left join on region == country.
  - Report matched/unmatched regions.
"""
import pandas as pd

# File paths
main_dataset_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset_v3.csv'
latitude_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/country_latitude.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset_v4.csv'

# --- Load latitude data ---
print("Loading latitude data...")
df_lat = pd.read_csv(latitude_file)
print(f"Latitude data shape: {df_lat.shape}")

# Normalization helper
def normalize_name(name):
    return str(name).lower().strip()

# Build normalized lookup: norm_name -> original_name
lat_country_map = {normalize_name(c): c for c in df_lat['country'].unique()}
print(f"Unique latitude countries: {len(lat_country_map)}")

# --- Discover regions in main dataset ---
print("Loading unique regions from main dataset...")
unique_regions = pd.read_csv(main_dataset_file, usecols=['region'])['region'].unique()
print(f"Unique regions: {len(unique_regions)}")

# --- Build region -> latitude_country mapping ---
region_map = {}
matched = 0
unmatched_list = []

# Potential overrides if needed (can copy from join_climate if useful)
overrides = {
    'usa': 'United States',
    'uk': 'United Kingdom',
    'uae': 'United Arab Emirates',
    'korea': 'South Korea',
    'vietnam': 'Vietnam',
    'russia': 'Russian Federation', # Latitude often uses full formal names
    'iran': 'Iran (Islamic Republic of)',
    'bolivia': 'Bolivia (Plurinational State of)',
    'venezuela': 'Venezuela (Bolivarian Republic of)',
    'ireland': 'Republic of Ireland',
}

# The latitude.csv had simple names like "Vietnam", "Russia" but let's check mapping
# Actually 'latitude.csv' had "Russian Federation" usually or similar.
# Let's rely on normalization first.

for region in unique_regions:
    norm = normalize_name(region)
    
    # Check map directly
    if norm in lat_country_map:
        region_map[region] = lat_country_map[norm]
        matched += 1
        continue
        
    # Check overrides
    found_override = False
    if norm in overrides:
        target_norm = normalize_name(overrides[norm])
        if target_norm in lat_country_map:
            region_map[region] = lat_country_map[target_norm]
            matched += 1
            found_override = True
    
    if not found_override:
        region_map[region] = None
        unmatched_list.append(region)

print(f"Matched regions: {matched}/{len(unique_regions)}")
if unmatched_list:
    print(f"Unmatched regions: {unmatched_list}")

# --- Prepare mapping dataframe ---
mapping_df = pd.DataFrame(list(region_map.items()), columns=['region', 'join_country'])

# Rename latitude country column for join
df_lat = df_lat.rename(columns={'country': 'join_country'})

# --- Process in chunks ---
print("Joining latitude data...")
chunk_size = 500000
first_chunk = True

try:
    chunks = pd.read_csv(main_dataset_file, chunksize=chunk_size)
    for chunk in chunks:
        # Add join_country via mapping
        chunk = pd.merge(chunk, mapping_df, on='region', how='left')
        
        # Join with latitude data
        chunk_merged = pd.merge(chunk, df_lat, on='join_country', how='left')
        
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
    print(f"Error joining latitude data: {e}")
