# Set up

Virtual Environment: https://docs.python.org/3/library/venv.html

## Dependencies

`pip install -r requirements.txt`

## Data Set

1. Download sqlite dataset: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
2. Extract dataset into current directory
```
├──  Data
    ├── _variable_descriptions.csv
    └── FPA_FOD_20221014.sqlite
```
## Prep Data
- Move reformat_data.py into the same directory as FPA_FOD_20221014.sqlite
- Run it via command line with `python reformat_data.py`
- This will create `updated_fires_db.sqlite` the new reformatted data

## Modified k-means algorithm
This should not affect the current behaviour
- Create copy of original `_kmeans.py` in `.venv\Lib\site-packages\sklearn\clustering` or whatever the package is installed
- Replace file with `_kmeans.py` in root directory
- When instantiating `KMeans` algorithm, add parameter `custom=True`. e.g., `KMeans(n_clusters=8, custom=True)`
  - Available parameters and default values: `custom=True`, `alpha=1`, `dimensions=1`

# Contributors

| Name | Username |
|---|---|
| Bilal Ayoub | bilalAhmadAyoub |
| Jethro Swanson | Jethro-Swanson |
| Kate Walley | KatieCodess |
| Henry Wong | hwbit |
