# Set up

Virtual Environment: https://docs.python.org/3/library/venv.html

## Dependencies

`pip install -r requirements.txt`

## Data Set

1. Download sqlite dataset: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
2. Extract dataset into root directory

## Prep Data
- Ensure you have a copy of the base fire data named FPA_FOD_20221014.sqlite in the main directory (not just in /Data but in the same dir as all the other files and init_data.py)
- Open/edit the init_data.py file to change any values want/care about
- Run `python init_data.py`
- This will create `updated_fires_db.sqlite` the new reformatted data and then create and add weights based on the defined cultural weights.json file

## Modified k-means algorithm
This should not affect the current behaviour
- Create copy of original `_kmeans.py` in `.venv\Lib\site-packages\sklearn\clustering` or whatever the package is installed
- Replace file with `_kmeans.py` in root directory
- When instantiating `KMeans` algorithm, add parameter `custom=True` to use custom distance function.
  - Available (optional) parameters and default values: `custom=True`, `alpha=1`, `dimensions=2`
  - e.g., `KMeans(n_clusters=8, custom=True)` is equivalent to `KMeans(n_clusters=8, custom=True, alpha=1, dimensions=2)`

## Running Analysis/Generating output
- Run jupyter notebook and open up the analysis_notebook
- Run the first two boxes of code (put true in for the query_db value to get weighted values or false to get unweighted)
- Output will be put in the output folder, we currently run BOTH our updated algo and the regular one on all inputs so when doing unweighted you will get two outputs,
one regular (we want this one) and one modified/custom that isn't applicable, and vice versa for when doing weighted.

# Contributors

| Name | Username |
|---|---|
| Bilal Ayoub | bilalAhmadAyoub |
| Jethro Swanson | Jethro-Swanson |
| Kate Walley | KatieCodess |
| Henry Wong | hwbit |
