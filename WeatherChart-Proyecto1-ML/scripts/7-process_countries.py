"""
7-process_countries.py
Processes the countries.csv dataset to keep only relevant socioeconomic
features for the music genre prediction model.

Keeps: Country, Continent, Population, GDP_per_capita
Drops: Rank, ID, IMF_GDP, UN_GDP (redundant or irrelevant)
"""
import pandas as pd

# File paths
input_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/countries.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/country_economy.csv'

columns_to_keep = ['Country', 'Continent', 'Population', 'GDP_per_capita']

print("Loading countries.csv...")
df = pd.read_csv(input_file)
print(f"Original shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# Select relevant columns
df_filtered = df[columns_to_keep].copy()

# --- Data Quality Checks ---
print("\n--- Data Quality Report ---")

# 1. Check for nulls
null_counts = df_filtered.isnull().sum()
print(f"\nNull values per column:\n{null_counts}")

# 2. Check for duplicates
dupes = df_filtered.duplicated(subset=['Country'])
print(f"\nDuplicate countries: {dupes.sum()}")

# 3. Basic stats for numeric columns
print(f"\nPopulation stats:\n{df_filtered['Population'].describe()}")
print(f"\nGDP_per_capita stats:\n{df_filtered['GDP_per_capita'].describe()}")

# 4. Continent distribution
print(f"\nCountries per continent:\n{df_filtered['Continent'].value_counts()}")

# --- Normalize column names to lowercase for consistency ---
df_filtered.columns = [c.lower() for c in df_filtered.columns]

# Save
print(f"\nFinal shape: {df_filtered.shape}")
print(f"Final columns: {df_filtered.columns.tolist()}")
df_filtered.to_csv(output_file, index=False)
print(f"Saved to {output_file}")
