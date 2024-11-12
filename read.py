# %%

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine, text

import mpl_toolkits.mplot3d
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def run():
    data = query_db()
    do_kmean(data)
    

def do_kmean(data):
    
    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(data)
    X = data
    y = kmeans.labels_
    
    estimators = [
        ("k_means_iris_8", KMeans(n_clusters=8)),
        ("k_means_iris_3", KMeans(n_clusters=3)),
        ("k_means_iris_20", KMeans(n_clusters=20))

    ]

    fig = plt.figure(figsize=(20, 16))
    titles = ["8 clusters", "3 clusters", "20 clusters"]
    for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(float), edgecolor="k")

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.set_xlabel("Long")
        ax.set_ylabel("Lat")
        ax.set_zlabel("DOY")
        ax.set_title(title)

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.show()

        
    
            
#query db, return array
def query_db():
    engine = create_engine("sqlite:///Data/FPA_FOD_20221014.sqlite")

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
        SELECT LATITUDE, LONGITUDE, DISCOVERY_DOY FROM Fires 
        WHERE FIRE_YEAR = 2008 
        AND NWCG_CAUSE_CLASSIFICATION = 'Human'
        LIMIT 5000
    '''

    with engine.connect() as connection:
        result = connection.execute(text(
        query
            )
        )

        data = []
        for row in result:
            a = []
            for item in row:
                a.append(item)

            data.append(a)
        # print(data)
        x = np.array(data)
        # print(x)
    return x


    
if __name__ == '__main__':
    run()
    

# %%
