# %%

"Running script..."

'''
Two method ways to run analysis: See run()

Option 1: 
Run the query one and map it to both 2 and 3d.
Require input parameters

e.g.,    
data, keys = query_db()
do_kmean_2d(data, keys)
do_kmean_3d(data, keys)
    
Option 2:  
Call do_kmeans_xd() directly and edit the query within the function
Does not need parameters

e.g.,
do_kmean_3d()


'''

import pytz
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from datetime import datetime
from sqlalchemy import create_engine, text
from scipy import stats

import mpl_toolkits.mplot3d

from sklearn.cluster import KMeans

# Database
ORIGINAL_DB = "sqlite:///FPA_FOD_20221014.sqlite"
MODIFIED_DB = "sqlite:///updated_fires_db.sqlite"

# set winnipeg timezone object for fileout
WINNIPEG_TZ = pytz.timezone('America/Winnipeg')

# output analysis file
FILE_BASE = "_output.txt"


def run():
    # Option 1: 
    # Run the query one and map it to both 2 and 3d
    
    # data, keys = query_db()
    # do_kmean_2d(data, keys)
    # do_kmean_3d(data, keys)
    
    # Option 2:  
    # Call do_kmeans_xd() directly and edit the query within the function
    
    do_kmean_3d()
    do_kmean_2d()
    
    # Do heatmap
    # map()
    

def do_kmean_3d(data=None, keys=None): 
    if data is None:
        X, keys = query_db( 
                    '''
                    SELECT DISCOVERY_DOY, LATITUDE, LONGITUDE FROM Fires 
                    WHERE FIRE_YEAR = 2008 
                    AND NWCG_CAUSE_CLASSIFICATION = 'Human'
                    AND NOT NWCG_GENERAL_CAUSE = 'Missing data/not specified/undetermined'
                    LIMIT 5000
                    '''
                    )        
    else:
        X = data
        
    # This will add the cluster column to the end of the labels
    keys.append("Cluster")

    # init for number of clusters the graph should have
    estimators = [
        ("k_means_8_custom_3d", KMeans(n_clusters=8, random_state=0, custom=True, alpha=1, dimensions=3)),
        ("k_means_8_3d", KMeans(n_clusters=8, random_state=0)),
    ]
    
    # size of the graph
    fig = plt.figure(figsize=(20, 16))
    titles = ["8 clusters - custom", "8 clusters - regular"]
        
    # loop through each algorithm cluster
    for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)
        
        # run the kmeans algorithm
        est.fit(X)
        
        # array of labels, the cluster a point belongs to
        labels = est.labels_
        
        # quick analysis of the cluster
        cluster_count(name, labels)
        analyze_clusters(name, X, labels, keys)

        # draws the points
        ax.scatter(X[:, 2], X[:, 1], X[:, 0], c=labels.astype(float), edgecolor="k")

        # axis titles
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.set_xlabel("Long")
        ax.set_ylabel("Lat")
        ax.set_zlabel("DOY")
        ax.set_title(title)

    #show the plot
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    

def do_kmean_2d(data=None, keys=None): 
    if data is None:
        X, keys = query_db( 
                    '''
                    SELECT DISCOVERY_DOY, LATITUDE, LONGITUDE FROM Fires 
                    WHERE FIRE_YEAR = 2008 
                    AND NWCG_CAUSE_CLASSIFICATION = 'Human'
                    AND NOT NWCG_GENERAL_CAUSE = 'Missing data/not specified/undetermined'
                    LIMIT 5000
                    '''
                    )        
    else:
        X = data
    
    keys.append("Cluster")
    
    # init for number of clusters the graph should have
    estimators = [
        ("k_means_8_2d", KMeans(n_clusters=8, custom=True)),
    ]

    # size of the graph
    fig = plt.figure(figsize=(20, 16))
    titles = ["8 clusters - custom"]
    
    # loop through each algorithm cluster
    for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
        ax = fig.add_subplot(2, 2, idx+1)
        
        # run the kmeans algorithm
        est.fit(X)
        
        # array of labels, the cluster a point belongs to
        labels = est.labels_
        
        # quick analysis of the cluster
        cluster_count(name, labels)
        analyze_clusters(name, X, labels, keys)

        # draws the points
        ax.scatter(X[:, 2], X[:, 1], c=labels.astype(float), edgecolor="k")

        # axis titles
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_xlabel("Long")
        ax.set_ylabel("Lat")
        ax.set_title(title)

    #show the plot
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def map():
    # create dataframe with columns
    # ensure that columns match query
    year = 2019
    columns = ["LONGITUDE", "LATITUDE"]
    data = query_db(
            f'''
            SELECT LONGITUDE, LATITUDE FROM Fires 
            WHERE FIRE_YEAR = {year}
            AND NWCG_CAUSE_CLASSIFICATION = 'Human'
            AND NOT NWCG_GENERAL_CAUSE = 'Missing data/not specified/undetermined'
            '''
        )
    
    df = pd.DataFrame(data, columns=columns)
    
    fig = px.density_mapbox(df, 
                            lat = 'LATITUDE', 
                            lon = 'LONGITUDE', 
                            # z = 'DISCOVERY_DOY',
                            radius = 1,
                            center = dict(lat = 39.0000, lon = -98.0000),
                            zoom = 2.5,
                            mapbox_style = 'open-street-map')
    fig.show()
    
    data = query_db (
        f'''        
        SELECT STATE,COUNT(*) AS state_count FROM Fires
        WHERE NWCG_CAUSE_CLASSIFICATION = 'Human'
        AND NWCG_GENERAL_CAUSE != 'Missing data/not specified/undetermined'
        AND FIRE_YEAR = {year}
        GROUP BY STATE
        ORDER BY state_count DESC
        LIMIT 10;
        '''
    )
    
    list = data.tolist()
    print(f"State\tCount for {year}\n")
    for row in list:
        print(f'{row[0]}\t{row[1]}\n')


def analyze_clusters(name, list, labels, headers=None):
    # change to array
    list_array = np.array(list).tolist()
    labels_array = np.array(labels).tolist()
    
    # add the cluster to the array
    clustered_list = []
    for i in range(len(list_array)):
        list_array[i].append(labels_array[i])
        clustered_list.append(list_array[i])
    
    # data frame for computing
    df = pd.DataFrame(clustered_list, columns=headers)
    
    results = {}
    
    # Iterate through each column
    for column in df.columns:
        col_data = df[column]
        
        # Safety check: Ensure the column is numeric
        if pd.api.types.is_numeric_dtype(col_data):
            # Calculate statistics
            mean = col_data.mean()
            median = col_data.median()
            std_dev = col_data.std()
            mode = col_data.mode().tolist()  # Mode can have multiple values
            data_range = col_data.max() - col_data.min()
            percentiles = {
                '25th': np.percentile(col_data, 25),
                '50th (median)': np.percentile(col_data, 50),
                '75th': np.percentile(col_data, 75),
            }
            
            # Store in the results dictionary
            results[column] = {
                'Mean': mean,
                'Median': median,
                'Standard Deviation': std_dev,
                'Mode': mode,
                'Range': data_range,
                'Percentiles': percentiles
            }
        else:
            results[column] = {"NaN": "Column does not continue numbers"}
        
    current_time = datetime.now(WINNIPEG_TZ)
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    
    # Out put results
    with open(f'{name}_metrics_{formatted_time}_{FILE_BASE}', "w") as f:
        f.write("Note: Not all columns will have meaningful output\n")
        for column, stats in results.items():
            f.write(f"Column: {column}")
            for stat_name, value in stats.items():
                f.write(f"  {stat_name}: {value}\n")
            f.write("\n")
        f.close()
        

# count the number of clusters
def cluster_count(name, labels):
    labels_array = np.array(labels).tolist()
 
    cluster_count = [0 for _ in range(max(labels_array) + 1)]
    for item in labels_array:
        cluster_count[int(item)] += 1
    
    current_time = datetime.now(WINNIPEG_TZ)
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    
    with open(f'{name}_cluster_count_{formatted_time}_{FILE_BASE}', "w") as f:
        f.write("Cluster Count\n")
        for index, item in enumerate(cluster_count):
            line = f'{index+1}: {str(item)}\n'
            f.write(line)
        f.close()


def histogram():
    engine = create_engine(ORIGINAL_DB)

    query = '''
        SELECT DISCOVERY_DOY FROM Fires 
        AND NWCG_CAUSE_CLASSIFICATION = 'Human'
        AND NOT NWCG_GENERAL_CAUSE = 'Missing data/not specified/undetermined'
    '''

    with engine.connect() as connection:
        result = connection.execute(text(query))
        data = []      
        
        for row in result:
            data.append(row[0])        
        
    plt.hist(data, bins=367)
    plt.title("Histogram for year 2008")
    plt.show()


#query db
def query_db(query=None):
    engine = create_engine(ORIGINAL_DB)

    # Column titles of interest
    
    # NWCG_REPORTING_AGENCY, 
    # FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME
    # NWCG_CAUSE_CLASSIFICATION, NWCG_GENERAL_CAUSE
    # FIRE_SIZE, FIRE_SIZE_CLASS
    # LATITUDE, LONGITUDE, STATE, COUNTY

    if query is None:
        query = '''
            SELECT DISCOVERY_DOY, LATITUDE, LONGITUDE FROM Fires 
            WHERE FIRE_YEAR = 2008 
            AND NWCG_CAUSE_CLASSIFICATION = 'Human'
            AND NOT NWCG_GENERAL_CAUSE = 'Missing data/not specified/undetermined'
            LIMIT 5000
        '''

    with engine.connect() as connection:
        # query returns a list of tuples
        result = connection.execute(text(query))
        keys = list(result.keys())
        
        # convert to array of array
        data = []
        for row in result:
            a = []
            for item in row:
                a.append(item)
            data.append(a)

        # usable form
        # [ [value1 value2. ] ... ]
        x = np.array(data)
    return x, keys

if __name__ == '__main__':
    run()
    

# %%
