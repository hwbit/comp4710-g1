# %%
import sqlite3
import numpy as np
from sqlalchemy import create_engine, text
import shutil



            
# create updated formatted db
def update_db():
    engine_base = create_engine("sqlite:///base_fires_db.sqlite") # original db
    shutil.copyfile('base_fires_db.sqlite', 'updated_fires_db.sqlite') # creates new updated db 
    engine_updated = create_engine("sqlite:///updated_fires_db.sqlite") # updated db
    
    
    
    
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



    
    # Gets all states
    states = []
    with engine_base.connect() as base_connection: # connects to the original db
        with engine_updated.connect() as updated_connection: # connects to the new updated db
              
            query = '''
                SELECT DISTINCT STATE FROM Fires
            '''     
             
            result = base_connection.execute(text(
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
            states = np.array(data)
            # print(x)

            #sqlite conn for error curcumvention
            sqliteConnection = sqlite3.connect('updated_fires_db.sqlite')


            # For every state, make a new column for it and set that column to 1 for whatever state the row represents
            for state_arr in states:
                state = state_arr[0]

                query = "ALTER TABLE Fires ADD State_" + state + " REAL DEFAULT 0;" #adds column

                updated_connection.execute(text(query))
                updated_connection.commit()
            
            
                query = "PRAGMA table_info(Fires);"
                result = updated_connection.execute(text(query))
                
                data = []
                for row in result:
                    a = []
                    for item in row:
                        a.append(item)

                    data.append(a)
                # print(data)
                endData = np.array(data)
                
                print("Printinng PRAGMA result")
                for row in endData:
                    print(row)
                
                
                
                cursor = sqliteConnection.cursor()
                print("Connected to SQLite")

                
                """
                metadata = MetaData()
                table = Table('Fires', metadata, autoload_with=engine_updated)

                column_to_update = table.columns.get("State_" + state)
                if column_to_update is None:
                    raise ValueError(f"Column 'State_{state}' not found in the table")
                
                condition_col = getattr(table.c, "STATE")
                
                # Create an update statement
                stmt = update(table).where(condition_col == state).values({column_to_update: 1})

                updated_connection.execute(stmt)
                updated_connection.commit()
                """
                
                # Update all state columns to be 1 for the state they reside in
                query = "UPDATE Fires SET State_" + state + ' = 1 WHERE STATE = "' + state + '";'
                cursor.execute(query)
                #result = updated_connection.execute(text(
                #        query
                #    )
                #)
                
                cursor.close()
            '''
            # remove all unused columns
            drop_queries = [
                "ALTER TABLE Fires DROP COLUMN FOD_ID;",
                "ALTER TABLE Fires DROP COLUMN FPA_ID;",
                "ALTER TABLE Fires DROP COLUMN SOURCE_SYSTEM_TYPE;",
                "ALTER TABLE Fires DROP COLUMN SOURCE_SYSTEM;",
                "ALTER TABLE Fires DROP COLUMN NWCG_REPORTING_UNIT_ID;",
                "ALTER TABLE Fires DROP COLUMN NWCG_REPORTING_UNIT_NAME;",
                "ALTER TABLE Fires DROP COLUMN SOURCE_REPORTING_UNIT;",
                "ALTER TABLE Fires DROP COLUMN SOURCE_REPORTING_UNIT_NAME;",
                "ALTER TABLE Fires DROP COLUMN LOCAL_FIRE_REPORT_ID;",
                "ALTER TABLE Fires DROP COLUMN LOCAL_INCIDENT_ID;",
                "ALTER TABLE Fires DROP COLUMN FIRE_CODE;",
                "ALTER TABLE Fires DROP COLUMN FIRE_NAME;",
                "ALTER TABLE Fires DROP COLUMN ICS_209_PLUS_INCIDENT_JOIN_ID;",
                "ALTER TABLE Fires DROP COLUMN ICS_209_PLUS_COMPLEX_JOIN_ID;",
                "ALTER TABLE Fires DROP COLUMN MTBS_ID;",
                "ALTER TABLE Fires DROP COLUMN MTBS_FIRE_NAME;",
                "ALTER TABLE Fires DROP COLUMN COMPLEX_NAME;",
                "ALTER TABLE Fires DROP COLUMN NWCG_CAUSE_AGE_CATEGORY;",
                "ALTER TABLE Fires DROP COLUMN CONT_DATE;",
                "ALTER TABLE Fires DROP COLUMN CONT_DOY;",
                "ALTER TABLE Fires DROP COLUMN CONT_TIME;",
                "ALTER TABLE Fires DROP COLUMN FIRE_SIZE_CLASS;",
                "ALTER TABLE Fires DROP COLUMN OWNER_DESCR;",
                "ALTER TABLE Fires DROP COLUMN FIPS_CODE;",
                "ALTER TABLE Fires DROP COLUMN FIPS_NAME;",
            ]
            
            for query in drop_queries:
                pass
            
            query = "ALTER TABLE Fires DROP COLUMN FOD_ID, FPA_ID, SOURCE_SYSTEM_TYPE, SOURCE_SYSTEM, NWCG_REPORTING_UNIT_ID, NWCG_REPORTING_UNIT_NAME, SOURCE_REPORTING_UNIT, SOURCE_REPORTING_UNIT_NAME, LOCAL_FIRE_REPORT_ID, LOCAL_INCIDENT_ID, FIRE_CODE, FIRE_NAME, ICS_209_PLUS_INCIDENT_JOIN_ID, ICS_209_PLUS_COMPLEX_JOIN_ID, MTBS_ID, MTBS_FIRE_NAME, COMPLEX_NAME, NWCG_CAUSE_AGE_CATEGORY, CONT_DATE, CONT_DOY, CONT_TIME, FIRE_SIZE_CLASS, OWNER_DESCR, FIPS_CODE, FIPS_NAME;"
                
            result = updated_connection.execute(text(
                    query
                )
            )
            '''
            
            # Get top results from these changes
            
            query = "SELECT * FROM Fires LIMIT 10"
            
            result = updated_connection.execute(text(
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
            endData = np.array(data)
            
            return endData
    
if __name__ == '__main__':
    data = update_db()
    for row in data:
        print("New Item")
        for item in row:
            print(item)

# %%
