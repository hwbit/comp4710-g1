# Set up
set up a virtual environment (recommended)
Virtual Environment: https://docs.python.org/3/library/venv.html

## Dependencies

`pip install -r requirements.txt`

## Data Set

1. Download sqlite dataset: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
2. Extract dataset into root directory

## Prep Data
- Ensure you have a copy of the base fire data named FPA_FOD_20221014.sqlite in the main directory  in the same dir as all the other files and init_data.py)
- Open/edit the init_data.py file to change any values want/care about in the constants (if this is the first time running this make sure the reformat_data update_db is uncommented, as this is what first sets up and normalizes the data)[This will take a while]
- Remember to correctly define which weights json file you wish to use for populating the cultural weights `Daylight_Saving_Weights.json` or `holiday_weights.json`
- Run `python init_data.py`
- This will create `updated_fires_db.sqlite` the new reformatted data and then create and add weights based on the defined cultural weights.json file

## Modified k-means algorithm
This should not affect the current behaviour
- Create copy of original `_kmeans.py` in `.venv\Lib\site-packages\sklearn\clustering` or whereever the sklearn package is installed
- Replace file with `_kmeans.py` in root directory
- When instantiating `KMeans` algorithm, add parameter `custom=True` to use custom distance function.
  - Available (optional) parameters and default values: `custom=True`, `alpha=1`, `dimensions=2`
  - e.g., `KMeans(n_clusters=8, custom=True)` is equivalent to `KMeans(n_clusters=8, custom=True, alpha=1, dimensions=2)`

## Running Analysis/Generating output
- Run `jupyter notebook` and open up the "analysis_notebook"
- Run the first cell to set up the db query function
- Run the second cell with an input of `True` to the function if you wish to use cultural weights or `False` if you do not
- Run the third (final) cell to do the clustering
- Output will be put in the output/ folder. Two sub-folders will be generated, one for the custom modified algorithm and one using base kmeans clustering
- Files will have heatmaps, an xml file of frequent files found, and a txt file of statistics about the clusters such as average values or counts of specific columns.

# Contributors

| Name | Username |
|---|---|
| Bilal Ayoub | bilalAhmadAyoub |
| Jethro Swanson | Jethro-Swanson |
| Kate Walley | KatieCodess |
| Henry Wong | hwbit |
