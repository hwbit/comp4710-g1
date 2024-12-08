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
import os
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# might be necessary for older package libraries
import mpl_toolkits.mplot3d

# CONSTANTS
DB_MODIFIED = "sqlite:///updated_fires_db.sqlite"
DB_ORIGINAL = "sqlite:///FPA_FOD_20221014.sqlite"
CLUSTER_COLUMN = "Cluster"
TEXT_FILE_BASE = "output.txt"
EXCEL_FILE_BASE = "output.xlsx"

WINNIPEG_TZ = pytz.timezone('America/Winnipeg')

APRIORI_ITEMSETS = "itemsets"
APRIORI_LENGTH = "length"
APRIORI_OCCURRENCE = "occurrence"
APRIORI_SUPPORT = "support"

# OPTIONS
APRIORI_MINSUP = 0.01  # Set to extremely low number to get more itemsets e.g., 0.0000001

INTEREST_COLUMNS = ["DISCOVERY_DOY", "STATE", "NWCG_GENERAL_CAUSE", "FIRE_SIZE_CLASS"] # Columns we are interested on finding frequent patterns
HEATMAP_COLUMNS = ["LATITUDE", "LONGITUDE"]

# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#############################
# Application flow
#############################

def run():
    # Column titles of interest
    
    # FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME
    # NWCG_GENERAL_CAUSE, NWCG_REPORTING_UNIT_NAME
    # FIRE_SIZE_CLASS
    # LATITUDE, LONGITUDE, STATE, COUNTY
    
    # Query string for the database
    
    SELECT_ARRAY = ["DISCOVERY_DOY", "LATITUDE", "LONGITUDE", "FIRE_YEAR", "STATE", "NWCG_GENERAL_CAUSE", "FIRE_SIZE_CLASS", "NWCG_REPORTING_UNIT_NAME"]
    select = ", ".join(map(str, SELECT_ARRAY))
    
    query = f'''
        SELECT {select} FROM Fires 
        WHERE NOT NWCG_GENERAL_CAUSE = 'Missing data/not specified/undetermined'
        LIMIT 1000
    '''
    
    # specify projection of the graph
    projection = "3d"

    # Do database call and run algorithms
    data, cleaned_data, column_headers = query_db(query=query, engine_str=DB_ORIGINAL)
    
    # order for the Minkownski distance
    order = len(cleaned_data[0])
    
    # Calculate alpha value for confounding variables from DOY
    # NOTE save value as constant for future runs
    # alpha = confounding_strength_DOY(data, column_headers)
    alpha = 1
    
    # # Do Clustering algorithm
    do_kmean(data, cleaned_data, column_headers, query, alpha, order, projection) #default is projection="2d"
    
    # Do heatmap
    # heatmap()
    

#############################
# Main functions
#############################

def do_kmean(data, cleaned_data, column_headers, query, alpha=1, order=2, projection="2d"):
    '''
    Run the kmeans algorithm and plot a 3d projection
    
    :param data: numpy array from sql search
    :param cleaned_data: numpy array from sql search without string columns
    :param column_headers: columns titles in the array search
    :param query: query string used - add context to the search
    :param order: order used for Minkowski distance
    :param projection: dimensions of the visual scatterplot
    '''
    
    X = data
    cleaned_x = cleaned_data
       
    # Add Cluster to column_headers
    column_headers.append(CLUSTER_COLUMN)

    # init k-means clusters and extra params
    # param custom indicates using custom distance formula
    estimators = [
        (f"k_means_8_custom_{projection}", KMeans(n_clusters=8, random_state=0, custom=True, alpha=alpha, order=order)),
        (f"k_means_8_{projection}", KMeans(n_clusters=8, random_state=0)),
    ]
    # title for the graphs, should be in order as the estimators
    titles = ["8 clusters - custom", 
              "8 clusters - regular"
              ]
    
    # size of the graph
    fig = plt.figure(figsize=(20, 16))

    # loop through each algorithm cluster
    for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
        # Time info to write to file
        current_time = datetime.now(WINNIPEG_TZ)
        formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
        
        # Create folders for the outputs
        file_name = f'{name}_{formatted_time}'
        os.mkdir(f'output/{file_name}')
        
        if projection == "3d":
            ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)
        else:
            ax = fig.add_subplot(2, 2, idx+1)

        # run the kmeans algorithm
        est.fit(cleaned_x)
        
        # array of labels, the cluster a point belongs to
        labels = est.labels_
                
        # add cluster to data and get DataFrame
        df = convert_data_to_dataframe(X, labels, column_headers)
        
        # analysis of the algorithm
        analyze_clusters(name, df, query, file_name)

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
    plt.savefig(f"output/{titles[0]}_{titles[1]}.png")
    plt.show()


def heatmap_by_year():
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

def confounding_strength_DOY(data, column_headers, confounder_column='DISCOVERY_DOY'):
    '''
    Calculating the confounding strength for DISCOVERY_DOY
    
    :param data: raw data
    :param column_headers: headers for raw data
    :param confounder_column: column to find confounding strength on
    
    :returns: confounding strength
    '''
    df = pd.DataFrame(data, columns=column_headers)
    # Converting all non-numeric values to NaN and dropping the columns
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    numeric_columns = df_numeric.select_dtypes(include='number')
    df_no_nan_any = df_numeric.dropna(axis=1, how='any')

    # Calculate correlations between 'DOY' and all other columns
    correlations = [
        df_no_nan_any[confounder_column].corr(df_no_nan_any[column]) 
        for column in df_no_nan_any if column != confounder_column
    ]
    
    # Compute average absolute correlation as confounding strength
    confounding_strength = np.mean(np.abs(correlations))
    
    print(f"Confounding Strength for {confounder_column}': {confounding_strength}")
    return confounding_strength


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
    print("HEADERS: " + str(headers))
    df = pd.DataFrame(clustered_list, columns=headers)
    
    return df


def analyze_clusters(name, df, query, file_name):
    '''
    Run analysis on the output from the algorithm
    
    Analysis results will be saved in the /output dir
    
    :param name: title
    :param df: dataframe format of the data
    :param header: column titles
    '''
    
    # Get general overall non-clustered results
    overall_results, overall_apriori_df, overall_heatmap_df = build_results(df)

    # Count the number of clusters
    cluster_column = df[CLUSTER_COLUMN]
    cluster_count = cluster_column.value_counts()
    
    # Group by cluster
    clusters = df.groupby(cluster_column)
    
    # Create a dictionary to store fp results
    clustered_itemsets_results = {}
    
    # Do Apriori on entire set
    overall_itemsets = do_apriori(overall_apriori_df)
    clustered_itemsets_results['OVERALL'] = overall_itemsets
    
        
    with open(f'output/{file_name}/{file_name}_metrics_{TEXT_FILE_BASE}', "w") as f:
        # Remove white space from query string into array and join them together
        stripped_lines = [line.strip() for line in query.splitlines() if line.strip()]
        query_clean = "\n  ".join(stripped_lines)
        
        # Write query string for context
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
        for cluster_name, cluster_data in clusters:
            f.write(f"\nCluster: {cluster_name}\n")

            # Build the results for each cluster
            results, apriori_df, heatmap_df = build_results(cluster_data)                   
            
            # run arpiori on the cluster
            itemsets = do_apriori(apriori_df)
            clustered_itemsets_results[cluster_name] = itemsets
            
            # Output results for each cluster
            for column, stats in results.items():
                f.write(f"  Column: {column}\n")
                for stat_name, value in stats.items():
                    f.write(f"    {stat_name}: {value}\n")
                f.write("\n")
                
            heatmap(heatmap_df, cluster_name, file_name)
        
        # Write out the itemset to Excel file
        with pd.ExcelWriter(f'output/{file_name}/{file_name}_fp_{EXCEL_FILE_BASE}') as writer:
            # Write each DataFrame to a different sheet
            for item in clustered_itemsets_results:
                clustered_itemsets_results[item].to_excel(writer, sheet_name=f'Cluster_{item}', index=False)
            
        # Raw DataFrame - contains the results of everything
        # f.write("\nDataFrame\n")
        # f.write(df.to_string())
        
    f.close()


def heatmap(df, cluster_name, file_name):
    '''
    Create a heatmap for a given cluster
    
    :param df: DataFrame of the cluster
    :param cluster_name: claster name
    :param file_name: file name for writing
    
    '''
    fig = px.density_mapbox(df, 
                            lat = 'LATITUDE', 
                            lon = 'LONGITUDE', 
                            radius = 1,
                            center = dict(lat = 39.0000, lon = -98.0000),
                            zoom = 2.5,
                            mapbox_style = 'open-street-map',
                            title= f'{file_name}: Fire heatmap in the United States for {cluster_name}',
                            )

    fig.show()
    fig.write_html(f'output/{file_name}/{file_name}_cluster{cluster_name}_heatmap.html')


def build_results(data):
    '''
    Accepts dataset and collects results
    Builds a DataFrame collection for string values
    
    :param data: dataset of to run analysis on
    
    :returns: results, dataframe
    '''
    results = {} # store results for each individual cluster
    heatmap_df = pd.DataFrame()
    apriori_df = pd.DataFrame()
    
    for column in data:                     
        col_data = data[column] 
        
        # Add to clustered dataframe we want to run Apriori algorith on
        if column in INTEREST_COLUMNS:
            apriori_df[column] = col_data
        # Add to heatmap dataframe for visuals
        if column in HEATMAP_COLUMNS:
            heatmap_df[column] = col_data
        
        # Converts the column to numbers, strings will turn to NaN
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
            # Count the number of item a thing occurs
            value_counts = col_data.value_counts()
            results[column] = value_counts.to_dict()
            
    return results, apriori_df, heatmap_df


def do_apriori(df):
    '''
    Do Apriori Algorith on DataFrame
    
    Filters itemsets of length 1
    
    :param df: dataframe of interest
    "returns: dataframe of itemsets (support, itemsets, length, and occurrence)
    '''
    te = TransactionEncoder()
    
    # convert df to an array and transform it to T/F arrays
    te_array = te.fit_transform(df.to_numpy())
    df_transformed = pd.DataFrame(te_array, columns=te.columns_)
    
    total_transactions = len(df_transformed)

    frequent_itemsets = apriori(df_transformed, min_support=APRIORI_MINSUP, use_colnames=True)
    
    # Modify and filter out columns of length 1
    frequent_itemsets[APRIORI_LENGTH] = frequent_itemsets[APRIORI_ITEMSETS].apply(lambda x: len(x))
    frequent_itemsets[APRIORI_ITEMSETS] = frequent_itemsets[APRIORI_ITEMSETS].apply(lambda x: ', '.join(map(str, x)))
    frequent_itemsets[APRIORI_OCCURRENCE] = (frequent_itemsets[APRIORI_SUPPORT] * total_transactions).astype(int)
    frequent_itemsets_filtered = frequent_itemsets[frequent_itemsets[APRIORI_LENGTH] != 1]

    return frequent_itemsets_filtered


def query_db(query, engine_str = DB_ORIGINAL):
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
