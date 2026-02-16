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
