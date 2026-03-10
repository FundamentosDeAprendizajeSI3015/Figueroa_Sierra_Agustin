import pandas as pd
import os

# File paths
input_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/GlobalLandTemperaturesByCountry.csv'
output_file = 'd:/Clase-fundamentos-aprendizaje-automatico/WeatherChart/data/country_monthly_temps.csv'

print("Loading climate data...")
try:
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    
    # Convert dt to datetime
    df['dt'] = pd.to_datetime(df['dt'])
    
    # Filter for data from 1970 onwards
    print("Filtering data from 1970 onwards...")
    df_filtered = df[df['dt'].dt.year >= 1970].copy()
    print(f"Shape after date filtering: {df_filtered.shape}")
    
    # Drop rows with missing AverageTemperature
    df_filtered = df_filtered.dropna(subset=['AverageTemperature'])
    print(f"Shape after dropping null temps: {df_filtered.shape}")
    
    # Extract month
    df_filtered['month'] = df_filtered['dt'].dt.month
    
    # Group by Country and Month, calculate mean temperature
    print("Calculating monthly averages per country...")
    # We group by Country and Month to get the 12-month profile for each country
    # We take the mean of 'AverageTemperature'
    df_monthly_avg = df_filtered.groupby(['Country', 'month'])['AverageTemperature'].mean().reset_index()
    
    # Rename columns for clarity in join
    df_monthly_avg.columns = ['country', 'month', 'avg_temp']
    
    print(f"Resulting shape (Countries * Months): {df_monthly_avg.shape}")
    print("Sample rows:")
    print(df_monthly_avg.head(12))
    
    # Validate consistency (check a few countries have 12 months)
    counts = df_monthly_avg['country'].value_counts()
    print(f"Countries with 12 months of data: {sum(counts == 12)} / {len(counts)}")
    
    print(f"Saving to {output_file}...")
    df_monthly_avg.to_csv(output_file, index=False)
    print("Done.")

except Exception as e:
    print(f"Error processing climate data: {e}")
