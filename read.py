# %%

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine, text

# allows reading pf pyx files
import pyximport; pyximport.install()

import mpl_toolkits.mplot3d
import numpy as np

from sklearn.cluster import KMeans

def run():
    data = query_db()
    do_kmean_3d(data)
    

def do_kmean_3d(data):
    X = data

    # init for number of clusters the graph should have
    estimators = [
        ("k_means_8", KMeans(n_clusters=8, custom=True)),
        ("k_means_3", KMeans(n_clusters=8)),
    ]

    # size of the graph
    fig = plt.figure(figsize=(20, 16))
    titles = ["8 clusters - custom", "8 clusters - regular"]
    
    # loop through each algorithm cluster
    for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)
        
        # run the kmeans algorithm
        est.fit(X)
        
        # array of labels
        # the cluster a point belongs to
        labels = est.labels_

        # draws the points
        ax.scatter(X[:, 1], X[:, 2], X[:, 0], c=labels.astype(float), edgecolor="k")

        # axis titles
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.set_xlabel("Lat")
        ax.set_ylabel("Long")
        ax.set_zlabel("DOY")
        ax.set_title(title)

    #show the plot
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    

    
#query db, return array
def query_db():
    engine = create_engine("sqlite:///Data/FPA_FOD_20221014.sqlite")

    # Column titles of interest
    
    # NWCG_REPORTING_AGENCY
    # FIRE_YEAR
    # DISCOVERY_DATE
    # DISCOVERY_DOY
    # DISCOVERY_TIME
    # NWCG_CAUSE_CLASSIFICATION
    # NWCG_GENERAL_CAUSE
    # FIRE_SIZE
    # FIRE_SIZE_CLASS
    # LATITUDE
    # LONGITUDE
    # STATE
    # COUNTY

    query = '''
        SELECT DISCOVERY_DOY, LATITUDE, LONGITUDE FROM Fires 
        WHERE FIRE_YEAR = 2008 
        AND NWCG_CAUSE_CLASSIFICATION = 'Human'
        LIMIT 5000
    '''

    with engine.connect() as connection:
        # query returns a list of tuples
        result = connection.execute(text(query))
        
        # convert to array of array
        data = []
        for row in result:
            a = []
            for item in row:
                a.append(item)

            data.append(a)
        # print(data)
        
        # usable form
        # [ [value1 value2. ] ... ]
        x = np.array(data, "int")
        # print(x)
    return x


# histogram
def query_db_2():
    engine = create_engine("sqlite:///Data/FPA_FOD_20221014.sqlite")

    query = '''
        SELECT DISCOVERY_DOY FROM Fires 
        WHERE FIRE_YEAR = 2008 
        AND NWCG_CAUSE_CLASSIFICATION = 'Human'
    '''

    with engine.connect() as connection:
        # query returns a list of tuples
        result = connection.execute(text(query))

        data = []
                    
        for row in result:
            data.append(row[0])        
        
    plt.hist(data, bins=367)
    plt.title("Histogram for year 2008")
    plt.show()
        
    return data


if __name__ == '__main__':
    run()
    

# %%
