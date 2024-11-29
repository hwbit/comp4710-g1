
# ADD column cultural relevence weight
# ADD weights, one holiday at a time -> find day with the holiday -> add full value there
# then go up or down X number of indicies filling in weight values for them (this would work in pandas)
# profit

import pandas as pd
import os
import json
from sqlalchemy import create_engine, text

DATE_SQL_FILE = "insert_dates.sql"
HOLIDAY_WEIGHTS = "holiday_weights.json"

DECAY = 0.5 # amount a cultural weight decreeases for each day away from the holiday
MIN_CULTURAL_WEIGHT_APPLIED = 0.001 # Smallest cultural weight that can be added to a day from a holiday

min_cultural_weight = 0 
max_cultural_weight = 0

# Normalizes weights  based on the min and max values found later
def normalize(weight):
    return (weight - min_cultural_weight)/(max_cultural_weight - min_cultural_weight)

# Creates a Calendar table in the updated sqlite db
def create_weighted_table():
    global min_cultural_weight
    global max_cultural_weight
    
    # Should be a connection to the newly updated sql file
    engine_updated = create_engine("sqlite:///updated_fires_db.sqlite") # updated db
        
    with engine_updated.connect() as updated_connection: # connects to the new updated db
        # Allows easy removal of population of base date data
        if(True):
            # Remove prevoius table so we can do a fresh creation
            query = "DROP TABLE IF EXISTS Calendar;" 
            updated_connection.execute(text(query))
            updated_connection.commit()
            
            # Create The Cultural_Dates table
            query = "CREATE TABLE Calendar (DOY Integer, Year Integer, Date String, Description String);"
            updated_connection.execute(text(query))
            updated_connection.commit()
            
            # adds all generated date entries to the table
            with open(DATE_SQL_FILE) as file:
                for query in file:
                    updated_connection.execute(text(query))
                    updated_connection.commit()
            
        # Gets the dates as a df cd
        query = "Select * from Calendar"
        date_df = pd.read_sql(query, updated_connection)
        print(date_df.head())
        
        # Gets the weights of holidays
        with open(HOLIDAY_WEIGHTS) as weight_file:
            holiday_weights = json.load(weight_file)

        # Add cutural relevence weights to the df
        date_df["Date_weight"] = 0
        
        # Look at every date to find holidays
        for index in range(0,len(date_df.index)):
            row_desc = date_df.loc[index, 'Description']
            
            # If the day is a holiday
            if(row_desc in holiday_weights.keys()):
                base_weight = holiday_weights[row_desc] # Get cultural weight of that holiday
                
                date_df.loc[index, 'Date_weight'] += base_weight
                
                # Traverses days *before* the holiday adding the decreased cultural weight to them
                decreased_weight = base_weight * DECAY
                curr_index = index - 1
                while(decreased_weight > MIN_CULTURAL_WEIGHT_APPLIED and curr_index>=0):
                    date_df.loc[curr_index, 'Date_weight'] += decreased_weight
                    curr_index = curr_index - 1
                    decreased_weight = decreased_weight * DECAY
                
                # Traverses days *after* the holiday adding the decreased cultural weight to them  
                decreased_weight = base_weight * DECAY
                curr_index = index + 1
                while(decreased_weight > MIN_CULTURAL_WEIGHT_APPLIED and curr_index<len(date_df.index)):
                    date_df.loc[curr_index, 'Date_weight'] += decreased_weight
                    curr_index = curr_index + 1
                    decreased_weight = decreased_weight * DECAY    
        
        min_cultural_weight = date_df['Date_weight'].min()
        max_cultural_weight = date_df['Date_weight'].max()

        # Normalize the cultural weights to be between 0 and 1
        date_df['Date_weight'] = date_df['Date_weight'].apply(normalize)
        
        date_df.set_index('Date', inplace=True) # Remove the index column to allow it to be put into the db
        
        # Removes current table so we can use our own updated version of it with weights
        query = "DROP TABLE Calendar;"
        updated_connection.execute(text(query))
        updated_connection.commit()
        
        # Adds the weights back to the db
        date_df.to_sql('Calendar', con=engine_updated)
        
        # Gets the dates as a df cd
        query = "Select * from Calendar LIMIT 500"
        updated_df = pd.read_sql(query, updated_connection)
        with open('headOutput.txt', 'w') as file:
            pd.options.display.max_rows = 1000
            file.write(str(updated_df.head(366)))
        
        updated_connection.commit()
        #Add in the weights to a new column of cultural relevence weights
    
create_weighted_table()