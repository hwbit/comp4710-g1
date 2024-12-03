# %%
import init_data_reformat_data
import init_data_create_sql
import init_data_add_cultural_relevence

WEIGHT_DECAY = 0.5 # Value multiplied to holiday weight for every day its away from the actual holiday, lower = faster decay
MIN_WEIGHT_APPLIED = 0.001 #Smallest weight applied to a day, smaller weights are ignored, prevents unessesasry additional 
WEIGHTS_JSON = "holiday_weights.json" # File with json formatted holiday:weight entries         
            
if __name__ == '__main__':
    # Normalizes data and converts non numeric columns into numeric by expanding each element in them to a new unique column
    # NOTE: This takes a long time, and only needs to be done once as date creation is seperate, feel free to comment out if you have already
    #   generated updated_fires_db.sqlite and just run the other two faster scripts
    #init_data_reformat_data.update_db()
    # Creates sql script to populate basic date and holiday info
    init_data_create_sql.main()
    # Creates a new 'Calendar' table in the db with the prev sql script and populates in with weights found in the weights json
    init_data_add_cultural_relevence.create_weighted_table(WEIGHT_DECAY, MIN_WEIGHT_APPLIED, WEIGHTS_JSON)
# %%
