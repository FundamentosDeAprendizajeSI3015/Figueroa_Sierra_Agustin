"""
9-process_latitude.py
Processes latitude.csv to extract geographic and socioeconomic data.

Cols to keep:
- Countries and areas -> country
- Latitude -> latitude
- Longitude -> longitude
- Gross_Tertiary_Education_Enrollment -> tertiary_enrollment
- Unemployment_Rate -> unemployment_rate

Handles encoding issues (cp1252/latin1) and checks for nulls.
"""
import pandas as pd
import numpy as np

input_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/latitude.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/country_latitude.csv'

# Columns mapping
cols_map = {
    'Countries and areas': 'country',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'Gross_Tertiary_Education_Enrollment': 'tertiary_enrollment',
    'Unemployment_Rate': 'unemployment_rate'
}

print("Loading latitude.csv...")
# Try reading with different encodings
try:
    df = pd.read_csv(input_file, encoding='cp1252')
except UnicodeDecodeError:
    print("cp1252 failed, trying latin1...")
    df = pd.read_csv(input_file, encoding='latin1')

print(f"Original shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# Strip whitespace from column names
df.columns = df.columns.str.strip()
print(f"Cleaned columns: {df.columns.tolist()}")

# Check if all needed columns exist
missing_cols = [c for c in cols_map.keys() if c not in df.columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols}")
    # Fallback or exit? For now exit as these are required.
    exit(1)

# Select and rename
df_filtered = df[list(cols_map.keys())].copy()
df_filtered = df_filtered.rename(columns=cols_map)

# ============================================================
# FIX: latitude.csv stores ALL coordinates as absolute values.
# We must apply negative signs for Southern and Western Hemisphere.
# ============================================================
print("\n--- Applying Hemisphere Sign Corrections ---")

# Countries that should have NEGATIVE LATITUDE (Southern Hemisphere)
SOUTH_HEMISPHERE = {
    'Argentina', 'Australia', 'Bolivia', 'Botswana', 'Brazil', 'Burundi',
    'Chile', 'Comoros', 'DR Congo', 'Democratic Republic of the Congo',
    'Ecuador', 'East Timor', 'Eswatini', 'Fiji', 'Indonesia',
    'Kenya', 'Lesotho', 'Madagascar', 'Malawi', 'Mauritius', 'Mozambique',
    'Namibia', 'Nauru', 'New Zealand', 'Papua New Guinea', 'Paraguay',
    'Peru', 'Rwanda', 'Samoa', 'Solomon Islands', 'South Africa',
    'Tanzania', 'Tonga', 'Tuvalu', 'Uganda', 'Uruguay', 'Vanuatu',
    'Zambia', 'Zimbabwe'
}

# Countries that should have NEGATIVE LONGITUDE (Western Hemisphere)
WEST_HEMISPHERE = {
    'Antigua and Barbuda', 'Argentina', 'Aruba', 'Bahamas', 'The Bahamas',
    'Barbados', 'Belize', 'Bolivia', 'Brazil', 'Canada', 'Chile',
    'Colombia', 'Costa Rica', 'Cuba', 'Curaçao', 'Dominica',
    'Dominican Republic', 'Ecuador', 'El Salvador', 'Grenada',
    'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Iceland', 'Ireland',
    'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru',
    'Portugal', 'Puerto Rico', 'Saint Kitts and Nevis', 'Saint Lucia',
    'Saint Vincent and the Grenadines', 'Spain', 'Suriname',
    'Trinidad and Tobago', 'United Kingdom', 'United States', 'Uruguay',
    'Venezuela'
}

# Apply sign corrections using normalized matching
south_fixed = 0
west_fixed = 0

for idx, row in df_filtered.iterrows():
    country = str(row['country']).strip()
    
    if country in SOUTH_HEMISPHERE and row['latitude'] > 0:
        df_filtered.at[idx, 'latitude'] = -abs(row['latitude'])
        south_fixed += 1
    
    if country in WEST_HEMISPHERE and row['longitude'] > 0:
        df_filtered.at[idx, 'longitude'] = -abs(row['longitude'])
        west_fixed += 1

print(f"  Latitudes corrected to negative (South): {south_fixed}")
print(f"  Longitudes corrected to negative (West): {west_fixed}")

# Validation: spot-check known coordinates
validation_checks = {
    'Argentina': (-38.4, -63.6),
    'Brazil': (-14.2, -51.9),
    'Australia': (-25.3, 133.8),
    'United States': (37.1, -95.7),
    'Colombia': (4.6, -74.3),
}
print("\n  Spot-check (expected vs actual):")
for country, (exp_lat, exp_lon) in validation_checks.items():
    row = df_filtered[df_filtered['country'] == country]
    if not row.empty:
        actual_lat = row['latitude'].values[0]
        actual_lon = row['longitude'].values[0]
        lat_ok = "✓" if (actual_lat < 0) == (exp_lat < 0) else "✗"
        lon_ok = "✓" if (actual_lon < 0) == (exp_lon < 0) else "✗"
        print(f"    {country:<20} lat={actual_lat:>8.2f} ({lat_ok})  lon={actual_lon:>8.2f} ({lon_ok})")

# Replace 0 with NaN for education/unemployment if 0 indicates missing data?
# Let's check stats first.
print("\n--- Data Quality Report ---")
print(df_filtered.describe())

# Check for nulls
print(f"\nNulls per column:\n{df_filtered.isnull().sum()}")

# Check for duplicates
print(f"\nDuplicate countries: {df_filtered.duplicated(subset=['country']).sum()}")

# Handle potential "0" values that should be NaN?
# Unemployment rate of 0 is unlikely unless specific small countries.
# Tertiary enrollment of 0 is also unlikely.
# Let's inspect rows with 0
zeros_unemp = df_filtered[df_filtered['unemployment_rate'] == 0]
print(f"\nCountries with 0 unemployment: {len(zeros_unemp)}")
if len(zeros_unemp) > 0:
    print(zeros_unemp.head())

# --- Manual Recovery for Missing Top Countries ---
# Hong Kong (400k rows in economy data) is missing from latitude.csv.
# Using approximations (Lat/Long real, others proxy from Singapore).
print("\nAdding missing Hong Kong data...")
hk_data = {
    'country': 'Hong Kong',
    'latitude': 22.3193,
    'longitude': 114.1694,
    'tertiary_enrollment': 84.8, # Proxy from Singapore
    'unemployment_rate': 4.11    # Proxy from Singapore
}
df_filtered = pd.concat([df_filtered, pd.DataFrame([hk_data])], ignore_index=True)

# Save
print(f"\nSaving to {output_file}...")
df_filtered.to_csv(output_file, index=False)
print("Done.")
