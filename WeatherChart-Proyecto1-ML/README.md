# Weather & Music Genre Prediction Project

This project predicts music genres based on environmental (climate, location) and socioeconomic factors. It integrates data from global music charts, climate records, and economic indicators to explore the relationship between our environment and the music we listen to.

## 📂 Project Structure

```
WeatherChart/
├── data/                   # Dataset files (charts, climate, economy, final datasets)
├── plots/                  # EDA visualizations (histograms, correlations, maps)
├── scripts/                # Python scripts for data processing and analysis
│   ├── 1-filter_charts.py      # Initial data filtering
│   ├── 3-join_datasets.py      # Smart artist-genre joining
│   ├── 10-process_latitude.py  # Latitude processing & HK recovery
│   ├── 12-eda.py               # Exploratory Data Analysis
│   └── ... (see full list below)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Recommended: High-RAM environment (16GB+) due to large datasets.

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/WeatherChart.git
    cd WeatherChart
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Data Pipeline
The project follows a linear pipeline (scripts `1-` to `12-`).

| Phase | Description | Key Scripts |
| :--- | :--- | :--- |
| **1. Preparation** | Filters raw ~3.5GB charts file. | `1-filter_charts.py` |
| **2. Integration** | Maps artists to genres using a "Smart Join" strategy. | `3-join_datasets.py` |
| **3. Climate** | Adds monthly avg temperatures (1970+). | `5-process_climate.py`, `6-join_climate.py` |
| **4. Socio/Geo** | Adds GDP, Population, Lat/Long. Recovered missing HK data. | `8-join_economy.py`, `10-join_latitude.py` |
| **5. Final Data** | Creates final training set from integrated data. | `11-create_training_set.py` |
| **6. Analysis** | Generates stats and visualizations. | `12-eda.py` |

## 📊 Key Insights (EDA)
- **Global Coverage**: 68 Countries.
- **Top Countries**: Costa Rica, Hong Kong, Brazil (by data volume).
- **Data Quality**: 100% complete (0 nulls) in critical columns for the final training set.

## 🔜 Next Steps
- **Modeling**: Training an XGBoost Classifier to predict `track_genre`.

---
*Created for Clase Fundamentos Aprendizaje Automatico.*
