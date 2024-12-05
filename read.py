# %%

"Running script..."

'''
Instructions:

Enter query in run()
Pass the results into the do_kmeans functions

'''

import numpy as np
import pytz
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px

from datetime import datetime
from sqlalchemy import create_engine, text

# might be necessary for older package libraries
import mpl_toolkits.mplot3d

from sklearn.cluster import KMeans

# Database
ORIGINAL_DB = "sqlite:///FPA_FOD_20221014.sqlite"
MODIFIED_DB = "sqlite:///updated_fires_db.sqlite"

# set winnipeg timezone object for fileout
WINNIPEG_TZ = pytz.timezone('America/Winnipeg')

# output analysis file
FILE_BASE = "_output.txt"


#############################
# Application flow
#############################

def run():
    # Column titles of interest
    
    # NWCG_REPORTING_AGENCY, 
    # FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME
    # NWCG_CAUSE_CLASSIFICATION, NWCG_GENERAL_CAUSE
    # FIRE_SIZE, FIRE_SIZE_CLASS
    # LATITUDE, LONGITUDE, STATE, COUNTY
    
    # Query string for the database
    query = '''
        SELECT DISCOVERY_DOY, LATITUDE, LONGITUDE, STATE FROM Fires 
        WHERE FIRE_YEAR = 2008 
        AND NWCG_CAUSE_CLASSIFICATION = 'Human'
        AND NOT NWCG_GENERAL_CAUSE = 'Missing data/not specified/undetermined'
        LIMIT 2000
    '''
    
    # specify dimensions of the graph
    dimensions = "3d"

    # Do database call and run algorithms
    data, cleaned_data, column_headers = query_db(query)
    do_kmean(data, cleaned_data, column_headers, query, dimensions) #default is dimensions="2d"

    # Do heatmap
    # map()


#############################
# Main functions
#############################

def do_kmean(data, cleaned_data, column_headers, query, dimensions="2d"):
    '''
    Run the kmeans algorithm and plot a 3d projection
    
    :param data: numpy array from sql search
    :param cleaned_data: numpy array from sql search without string columns
    :param column_headers: columns titles in the array search
    :param query: query string used - add context to the search
    :param dimensions: dimensions of the visual scatterplot
    '''
    
    X = data
    cleaned_x = cleaned_data
    
    # Add Cluster column to column_headers
    column_headers.append("Cluster")

    # init k-means clusters and extra params
    # param custom indicates using custom distance formula
    estimators = [
        (f"k_means_8_custom_{dimensions}", KMeans(n_clusters=8, random_state=0, custom=True, alpha=1, dimensions=2)),
        (f"k_means_8_{dimensions}", KMeans(n_clusters=8, random_state=0)),
    ]
    # title for the graphs, should be in order as the estimators
    titles = ["8 clusters - custom", 
              "8 clusters - regular"
              ]
    
    # size of the graph
    fig = plt.figure(figsize=(20, 16))

    # loop through each algorithm cluster
    for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
        if dimensions == "3d":
            ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)
        else:
            ax = fig.add_subplot(2, 2, idx+1)

        # run the kmeans algorithm
        est.fit(cleaned_x)
        
        # array of labels, the cluster a point belongs to
        labels = est.labels_
        
        # add cluster to data and get DataFrame
        df = convert_data_to_dataframe(X, labels, column_headers)
        
        # quick analysis of the cluster
        analyze_clusters(name, df, labels, query)

        # draws the points depending on
        # need to know the index of the column to plot the graph on
        if dimensions == "3d":
            ax.scatter(cleaned_x[:, 2], cleaned_x[:, 1], cleaned_x[:, 0], c=labels.astype(float), edgecolor="k")
        else:
            ax.scatter(cleaned_x[:, 2], cleaned_x[:, 1], c=labels.astype(float), edgecolor="k")

        # axis titles and axis labels
        ax.set_title(title)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_xlabel("Long")
        ax.set_ylabel("Lat")
        if dimensions == "3d":
            ax.zaxis.set_ticklabels([])
            ax.set_zlabel("DOY")

    #show the plot
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def heatmap():
    '''
    Draws a heatmap of the united states for fires for every year
    
    Show top 10 states
    '''
   
    for year in range(1992, 2021):
        data, _, keys = query_db(
                f'''
                SELECT LONGITUDE, LATITUDE FROM Fires 
                WHERE FIRE_YEAR = {year}
                AND NWCG_CAUSE_CLASSIFICATION = 'Human'
                AND NOT NWCG_GENERAL_CAUSE = 'Missing data/not specified/undetermined'
                '''
            )
        
        df = pd.DataFrame(data, columns=keys)
        
        fig = px.density_mapbox(df, 
                                lat = 'LATITUDE', 
                                lon = 'LONGITUDE', 
                                # z = 'DISCOVERY_DOY',
                                radius = 1,
                                center = dict(lat = 39.0000, lon = -98.0000),
                                zoom = 2.5,
                                mapbox_style = 'open-street-map',
                                title= f'Fire heatmap in the United States for {year}',
                                )
        fig.show()
        
        data, _, keys = query_db (
            f'''        
            SELECT STATE, COUNT(*) AS state_count FROM Fires
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



#############################
# Support functions
#############################

def convert_data_to_dataframe(list, labels, headers):
    '''
    Convert the the inputs to a dataframe
    
    :param list: numpy array of the inputs
    :param labels: list of cluster values
    :param headers: column titles
    
    :returns: data in dataframe format
    '''
    # change to array
    list_array = np.array(list).tolist()
    labels_array = np.array(labels).tolist()
    
    # Add the cluster to the array
    clustered_list = []
    for i in range(len(list_array)):
        list_array[i].append(labels_array[i])
        clustered_list.append(list_array[i])

    # data frame for computing
    df = pd.DataFrame(clustered_list, columns=headers)
    
    return df


def analyze_clusters(name, df, labels, query):
    '''
    Run analysis on the output from the algorithm
    
    Analysis results will be saved in the /output dir
    
    :param name: title
    :param df: dataframe format of the data
    :param header: column titles
    '''
    labels_array = np.array(labels).tolist()

    # Count the number of clusters
    cluster_count = [0 for _ in range(max(labels_array) + 1)]
    for item in labels_array:
        cluster_count[int(item)] += 1
    
    # Get general overall results for the cluster
    overall_results = analyze_results(df)
    
    # Create a dictionary to store results
    clustered_results = {}

    # Group by the values in the last column
    cluster_column = df.columns[-1]  # Get the name of the last column
    cluster = df.groupby(cluster_column)
    
    # Time info to write to file
    current_time = datetime.now(WINNIPEG_TZ)
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    
    with open(f'output/{name}_metrics_{formatted_time}_{FILE_BASE}', "w") as f:
        # Remove white space from query string into array and join them together
        stripped_lines = [line.strip() for line in query.splitlines() if line.strip()]
        query_clean = "\n  ".join(stripped_lines)
        
        f.write("Query String\n")
        f.write(f'  {query_clean}')
        f.write("\n")
        
        # Write the cluster count
        f.write("\nCluster Count\n")
        for index, item in enumerate(cluster_count):
            line = f'  {index}: {str(item)}\n'
            f.write(line)
        f.write("\n")
        
        # Overall results
        f.write("Overall Results\n")
        for column, stats in overall_results.items():
            f.write(f"  Column: {column}\n")
            for stat_name, value in stats.items():
                f.write(f"    {stat_name}: {value}\n")
            f.write("\n")
        
        # Iterate through each cluster
        for cluster_name, cluster_data in cluster:
            results = {}
            f.write(f"\nCluster: {cluster_name}\n")
            
            for column in cluster_data:
                col_data = cluster_data[column]
                
                # Converts the column to numbers, strings turn to NaN
                col_data_convert = pd.to_numeric(col_data, errors='coerce')

                # Ensure the column does not contain NaN
                if not col_data_convert.hasnans:
                    # Calculate statistics
                    mean = col_data_convert.mean()
                    median = col_data_convert.median()
                    std_dev = col_data_convert.std()
                    mode = col_data_convert.mode().tolist()  # Mode can have multiple values
                    data_range = col_data_convert.max() - col_data_convert.min()
                            
                    # Store in the results dictionary
                    results[column] = {
                        'Mean': mean,
                        'Median': median,
                        'Standard Deviation': std_dev,
                        'Mode': mode,
                        'Range': data_range,
                    }
                else:
                    value_counts = col_data.value_counts()
                    results[column] = value_counts.to_dict()         

                # Store the results for this group
                clustered_results[cluster_name] = results
                
            # Output results for each cluster
            for column, stats in results.items():
                f.write(f"  Column: {column}\n")
                for stat_name, value in stats.items():
                    f.write(f"    {stat_name}: {value}\n")
                f.write("\n")
        
        f.write("\nDataFrame\n")
        f.write(df.to_string())
    f.close()


def analyze_results(df: pd.DataFrame):
    '''
    Get non-clustered metrics for all columns
    
    :param df: DataFrame of the results
    :return: dictionary of non-clustered results
    '''
    
    results = {}
    
    # Iterate through each column
    for column in df.columns:
        col_data = df[column]
        
        # Converts the column to numbers, strings turn to NaN
        col_data_convert = pd.to_numeric(df[column], errors='coerce')

        # Ensure the column does not contain NaN
        if not col_data_convert.hasnans:
            # Calculate statistics
            mean = col_data_convert.mean()
            median = col_data_convert.median()
            std_dev = col_data_convert.std()
            mode = col_data_convert.mode().tolist()  # Mode can have multiple values
            data_range = col_data_convert.max() - col_data_convert.min()
            
            # Store in the results dictionary
            results[column] = {
                'Mean': mean,
                'Median': median,
                'Standard Deviation': std_dev,
                'Mode': mode,
                'Range': data_range,
            }
        else:
            value_counts = col_data.value_counts()
            results[column] = value_counts.to_dict()
            
    return results

#query db
def query_db(query, engine_str = ORIGINAL_DB):
    '''
    Call Database to run query
    
    :param query: sql query string
    :returns numpy, column_headers: numpy array, column headers
    '''
    engine = create_engine(engine_str)

    with engine.connect() as connection:
        # query returns a list of tuples
        result = connection.execute(text(query))
        column_headers = list(result.keys())
        
        # convert to query results to an array of an array
        # [ [value1 value2 ] [value1 value2 ] ... ]
        
        # First it iterates through each row and then each column in that row
        # if the column contains a numeric value, it will add to the raw and clean array
        # if the column contains string values, it will only be added to the raw array
        # then it is consolidated into the data, cleaned_data arrays
        # Reasoning: the clustering algorithm works best with numeric entries only
        # the raw array will be helpful for getting insights and grouping for those categories
        data = []
        cleaned_data = []
        for row in result:
            raw = []
            clean = []
            for column_item in row:
                raw.append(column_item)
                
                # Convert non-numeric to NaN                
                # Add to clean table if it is numeric
                numeric_item = pd.to_numeric(column_item, errors='coerce')
                if np.isfinite(numeric_item):
                    clean.append(column_item)
                    
            data.append(raw)
            cleaned_data.append(clean)

        # usable form

        x = np.array(data)
        cleaned_x = np.array(cleaned_data)
        
    return x, cleaned_x, column_headers

if __name__ == '__main__':
    run()

# %%
