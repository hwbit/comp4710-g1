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
- Ensure you have a copy of the base fire data named FPA_FOD_20221014.sqlite in the main directory (not just in /Data but in the same dir as all the other files and init_data.py)
- Open/edit the init_data.py file to change any values want/care about
- Run `python init_data.py`
- This will create `updated_fires_db.sqlite` the new reformatted data and then create and add weights based on the defined cultural weights.json file

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
