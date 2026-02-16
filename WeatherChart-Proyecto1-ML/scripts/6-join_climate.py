import pandas as pd
import os

# File paths
main_dataset_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset.csv'
climate_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/country_monthly_temps.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/final_dataset_v2.csv'

print("Loading datasets...")
# Read climate data
df_climate = pd.read_csv(climate_file)
print(f"Climate data loaded: {df_climate.shape}")

# Read main dataset (in chunks if huge, but we need to verify country names first)
# For the join, let's load unique regions first to build a map, then chunk process.
print("Loading unique regions from main dataset...")
unique_regions = pd.read_csv(main_dataset_file, usecols=['region'])['region'].unique()
print(f"Unique regions found: {len(unique_regions)}")

# Normalization helper
def normalize_name(name):
    return str(name).lower().strip()

# Build mapping from normalized climate country to original climate country
climate_country_map = {normalize_name(c): c for c in df_climate['country'].unique()}
print(f"Unique climate countries: {len(climate_country_map)}")

# Manual overrides for known discrepancies (can be expanded later)
overrides = {
    'usa': 'United States',
    'uk': 'United Kingdom',
    'uae': 'United Arab Emirates',
    'korea': 'South Korea', # Assuming typical k-pop context
    'vietnam': 'Vietnam', # check spelling
}

print("Mapping regions...")
region_join_map = {}
matched_count = 0
unmatched_list = []

for region in unique_regions:
    norm = normalize_name(region)
    
    # Try overrides first
    if norm in overrides and overrides[norm].lower() in climate_country_map:
        region_join_map[region] = overrides[norm] # map to whatever key matches normalize(climate_country)
        # Actually effectively we just need to match the 'country' column in climate df
        # But climate_country_map keys are normalized.
        # So: region_join_map[region] = climate_country_map[overrides[norm].lower()]
        target = overrides[norm]
        if normalize_name(target) in climate_country_map:
             region_join_map[region] = climate_country_map[normalize_name(target)]
             matched_count += 1
             continue

    # Try exact normalized match
    if norm in climate_country_map:
        region_join_map[region] = climate_country_map[norm]
        matched_count += 1
    else:
        unmatched_list.append(region)
        region_join_map[region] = None

print(f"Matched regions: {matched_count}/{len(unique_regions)}")
print(f"Unmatched regions (Top 10): {unmatched_list[:10]}")

# Prepare climate dataframe for merge
# It has columns: country, month, avg_temp
# We rename 'country' to 'join_country' to matching
df_climate = df_climate.rename(columns={'country': 'join_country'})

# Create mapping dataframe
mapping_df = pd.DataFrame(list(region_join_map.items()), columns=['region', 'join_country'])

print("Joining climate data...")
chunk_size = 500000
first_chunk = True

try:
    chunks = pd.read_csv(main_dataset_file, chunksize=chunk_size)
    for chunk in chunks:
        # 1. Add 'join_country' column
        chunk = pd.merge(chunk, mapping_df, on='region', how='left')
        
        # 2. Extract month
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk['month'] = chunk['date'].dt.month
        
        # 3. Join with climate data on join_country AND month
        chunk_merged = pd.merge(chunk, df_climate, on=['join_country', 'month'], how='left')
        
        # 4. Clean up
        chunk_merged = chunk_merged.drop(columns=['join_country'])
        
        # Save
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk_merged.to_csv(output_file, index=False, mode=mode, header=header)
        first_chunk = False
        print(".", end="", flush=True)

    print(f"\nJoin complete. Saved to {output_file}")
    
except Exception as e:
    print(f"Error joining climate data: {e}")
