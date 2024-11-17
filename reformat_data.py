# %%
import numpy as np
from sqlalchemy import create_engine, text
import shutil

# Convert search results to usable list
def conv_to_data(result):
    data = []
    for row in result:
        a = []
        for item in row:
            a.append(item)

        data.append(a)
    return np.array(data)


# converts all unique values from a column to new columns
def expand_to_columns(column, base_connection, updated_connection, desired_columns, prefix):
    # Gets all unique states from db
    query = "SELECT DISTINCT " + column + " FROM Fires;"    
    result = base_connection.execute(text(query))
    unique_col_values = conv_to_data(result)
    
        # For every value, make a new column for it and set that column to 1 for whatever value the row represents
    for value_arr in unique_col_values:
        value = value_arr[0]
        value = value.replace("/", "_") # Remove any /s that cause problems with making columns
        value = value.replace(" ", "_") # also remove spaces that will cause problems
        value = value.replace("&", "_") # also remove ampersands that will cause problems
        value = value.upper() # transitions to uppercase for duplicate checking
        
        # If this column is unique (there arn't multiples of the same name) create and pupulate it
        if(not(str(prefix+value) in desired_columns)):
            print("Adding " + prefix + value + " column")
            query = "ALTER TABLE Fires ADD " + prefix + value + " REAL DEFAULT 0;" #adds new column

            updated_connection.execute(text(query))
            updated_connection.commit()
            
            # Update all value columns to be 1 for what value they represent in the row
            query = "UPDATE Fires SET " + prefix + value + ' = 1 WHERE ' + column + ' = "' + value + '";'
            updated_connection.execute(text(query))
            updated_connection.commit()
        
            # Adds new columns to list of columns to keep
            desired_columns.append(prefix + value)
    
    return(desired_columns)


# create updated formatted db
def update_db():
    engine_base = create_engine("sqlite:///FPA_FOD_20221014.sqlite") # original db
    shutil.copyfile('FPA_FOD_20221014.sqlite', 'updated_fires_db.sqlite') # creates new updated db 
    engine_updated = create_engine("sqlite:///updated_fires_db.sqlite") # updated db
    
    # NWCG_REPORTING_AGENCY
    # FIRE_YEAR
    # DISCOVERY_DATE
    # DISCOVERY_DOY
    # DISCOVERY_TIME
    
    # Initial columns
    starting_columns = [
        "DISCOVERY_DOY",
        "FIRE_SIZE",
        "LATITUDE",
        "LONGITUDE",
    ]
    # List of columns we wish to have in the end result
    desired_columns = []
    desired_columns.append("OBJECTID") # Adds the primary key as a desired end column
    
    with engine_base.connect() as base_connection: # connects to the original db
        with engine_updated.connect() as updated_connection: # connects to the new updated db
              
            # dropping all TRIGGERs from schema to allow UPDATE commands
            query = "SELECT name FROM sqlite_master WHERE type = 'trigger';"
                
            result = updated_connection.execute(text(query))
            updated_connection.commit()
            
            # convert data to usable format
            trigger_names = conv_to_data(result)
            
            # delete each trigger
            for name in trigger_names:
                query = "DROP TRIGGER " + name[0] + ";"
                updated_connection.execute(text(query))
                updated_connection.commit()
            
            #convert column values to new columns
            if(True): 
                ########
                # STATES
                ########
                
                print("Converting states to columns")
                desired_columns = expand_to_columns("STATE", base_connection, updated_connection, desired_columns, "STATE_")
                
                ########
                # Reporting Agency
                ########
                
                print("Converting reporting agencies to columns")
                desired_columns = expand_to_columns("NWCG_REPORTING_AGENCY", base_connection, updated_connection, desired_columns, "AGENCY_")
            
                ########  
                # Cause
                ########
                
                print("Converting cause to columns")
                desired_columns = expand_to_columns("NWCG_GENERAL_CAUSE", base_connection, updated_connection, desired_columns, "CAUSE_")
            
        
            # Creates a list of existing columns to determine what columns need to be dropped
            print("Desired Columns: " + str(desired_columns))
            existing_columns = []
            query = "PRAGMA table_info(Fires);"
            result = updated_connection.execute(text(query))
            column_infos = conv_to_data(result)
            
            # For each column we get several parts of info that we must proccess
            for column_info in column_infos:
                #print("Row type: " + str(type(item)))
                #print("Item being processed as str: " + str(item))
                column_info_parts = str(column_info).split(" ") # seperate all these parts into individual things
                #print(item[1])
                existing_columns.append(column_info_parts[1]) # Select name of coulum from info to be added to existing columns
            
            ##########
            # Change table types
            ##########            
            
            print("Converting columns to REAL type")
            for old_col in starting_columns:
                print("Converrting: " + old_col)
                
                query = "ALTER TABLE Fires ADD COLUMN REAL_" + old_col + " REAL;"
                updated_connection.execute(text(query))
                updated_connection.commit()
    
                query = "UPDATE Fires SET REAL_" + old_col + " = CAST(" + old_col + " as REAL);"
                updated_connection.execute(text(query))
                updated_connection.commit()
                
                desired_columns.append("REAL_" + old_col)
                
            ########
            # DROPPING UNUSED COLUMNS
            ########
            print()
            print("Dropping Unused columns")
            # remove all unused columns
            for column_base in existing_columns:
                column = ''.join(c for c in column_base if c.isprintable()) # Attempts to remove any hidden charecters
                column = column.strip("'") # Removes extra quotations around a string if they exist
                
                # If the current column isn't one we want, drop it
                if(str(column) not in desired_columns):
                    print("Dropping Column: " + column)
                    updated_connection.execute(text("ALTER TABLE Fires DROP COLUMN " + column + ";"))
                    updated_connection.commit()
                    
            #########
            # Normalizing Columns 
            #########
            
            # Normalizes all columns we did not create ensuring they are also only 0 to 1
            print("Normalizing Columns")
            for column in starting_columns:
                print("Normalizing: REAL_" + column)
                
                # Get max value for the column
                query = "SELECT MAX(REAL_" + column + ") FROM Fires"
                result = updated_connection.execute(text(query))
                # Extract result from cursor
                for res in result:
                    columnMax = res[0]
            
                # Get min value for the column
                query = "SELECT MIN(REAL_" + column + ") FROM Fires"
                result = updated_connection.execute(text(query))
                for res in result:
                    columnMin = res[0]
                
                # Normalize the column to values from 0 to 1
                query = "UPDATE Fires Set REAL_" + column + " = (REAL_" + column + " - " + str(columnMin) + ")/" + str(float(columnMax) - float(columnMin)) + ";"
                updated_connection.execute(text(query))
                updated_connection.commit()
                
            # Get top results from these changes and what the columns look like
            query = "PRAGMA table_info(Fires);"
            result = updated_connection.execute(text(query))
            column_infos = conv_to_data(result)
            for column_info in column_infos:
                print(column_info)
            
            # selects a few lines to give examples of current data within the updated db
            query = "SELECT * FROM Fires LIMIT 5"
            result = updated_connection.execute(text(query))
            endData = conv_to_data(result)
            
            return endData
    
if __name__ == '__main__':
    data = update_db()
    for row in data:
        print("New Item:")
        print()
        for item in row:
            print(item)

# %%
