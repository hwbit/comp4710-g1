# Set up

Virtual Environment: https://docs.python.org/3/library/venv.html

## Dependencies

`pip install -r requirements.txt`

## Data Set

1. Download sqlite dataset: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
2. Extract dataset into current directory
```
├──  Data
    ├── _variable_descriptions.csv
    └── FPA_FOD_20221014.sqlite
```
## Prep Data
- Move reformat_data.py into the same directory as FPA_FOD_20221014.sqlite
- Run it via command line with `python reformat_data.py`
- This will create `updated_fires_db.sqlite` the new reformatted data

Columns in updated db:
[0 'OBJECTID' 'INTEGER' 1 None 1] (This is the primary key and is NOT normalized, this shouldn't be used in algos)
[1 'STATE_CA' 'REAL' 0 '0' 0]
[2 'STATE_NM' 'REAL' 0 '0' 0]
[3 'STATE_OR' 'REAL' 0 '0' 0]
[4 'STATE_NC' 'REAL' 0 '0' 0]
[5 'STATE_WY' 'REAL' 0 '0' 0]
[6 'STATE_CO' 'REAL' 0 '0' 0]
[7 'STATE_WA' 'REAL' 0 '0' 0]
[8 'STATE_MT' 'REAL' 0 '0' 0]
[9 'STATE_UT' 'REAL' 0 '0' 0]
[10 'STATE_AZ' 'REAL' 0 '0' 0]
[11 'STATE_SD' 'REAL' 0 '0' 0]
[12 'STATE_AR' 'REAL' 0 '0' 0]
[13 'STATE_NV' 'REAL' 0 '0' 0]
[14 'STATE_ID' 'REAL' 0 '0' 0]
[15 'STATE_MN' 'REAL' 0 '0' 0]
[16 'STATE_TX' 'REAL' 0 '0' 0]
[17 'STATE_FL' 'REAL' 0 '0' 0]
[18 'STATE_SC' 'REAL' 0 '0' 0]
[19 'STATE_LA' 'REAL' 0 '0' 0]
[20 'STATE_OK' 'REAL' 0 '0' 0]
[21 'STATE_KS' 'REAL' 0 '0' 0]
[22 'STATE_MO' 'REAL' 0 '0' 0]
[23 'STATE_NE' 'REAL' 0 '0' 0]
[24 'STATE_MI' 'REAL' 0 '0' 0]
[25 'STATE_KY' 'REAL' 0 '0' 0]
[26 'STATE_OH' 'REAL' 0 '0' 0]
[27 'STATE_IN' 'REAL' 0 '0' 0]
[28 'STATE_VA' 'REAL' 0 '0' 0]
[29 'STATE_IL' 'REAL' 0 '0' 0]
[30 'STATE_TN' 'REAL' 0 '0' 0]
[31 'STATE_GA' 'REAL' 0 '0' 0]
[32 'STATE_AK' 'REAL' 0 '0' 0]
[33 'STATE_ND' 'REAL' 0 '0' 0]
[34 'STATE_WV' 'REAL' 0 '0' 0]
[35 'STATE_WI' 'REAL' 0 '0' 0]
[36 'STATE_AL' 'REAL' 0 '0' 0]
[37 'STATE_NH' 'REAL' 0 '0' 0]
[38 'STATE_PA' 'REAL' 0 '0' 0]
[39 'STATE_MS' 'REAL' 0 '0' 0]
[40 'STATE_ME' 'REAL' 0 '0' 0]
[41 'STATE_VT' 'REAL' 0 '0' 0]
[42 'STATE_NY' 'REAL' 0 '0' 0]
[43 'STATE_IA' 'REAL' 0 '0' 0]
[44 'STATE_DC' 'REAL' 0 '0' 0]
[45 'STATE_MD' 'REAL' 0 '0' 0]
[46 'STATE_CT' 'REAL' 0 '0' 0]
[47 'STATE_MA' 'REAL' 0 '0' 0]
[48 'STATE_NJ' 'REAL' 0 '0' 0]
[49 'STATE_HI' 'REAL' 0 '0' 0]
[50 'STATE_DE' 'REAL' 0 '0' 0]
[51 'STATE_PR' 'REAL' 0 '0' 0]
[52 'STATE_RI' 'REAL' 0 '0' 0]
[53 'AGENCY_FS' 'REAL' 0 '0' 0]
[54 'AGENCY_BIA' 'REAL' 0 '0' 0]
[55 'AGENCY_TRIBE' 'REAL' 0 '0' 0]
[56 'AGENCY_BLM' 'REAL' 0 '0' 0]
[57 'AGENCY_NPS' 'REAL' 0 '0' 0]
[58 'AGENCY_BOR' 'REAL' 0 '0' 0]
[59 'AGENCY_FWS' 'REAL' 0 '0' 0]
[60 'AGENCY_ST_C_L' 'REAL' 0 '0' 0]
[61 'AGENCY_DOD' 'REAL' 0 '0' 0]
[62 'AGENCY_IA' 'REAL' 0 '0' 0]
[63 'AGENCY_DOE' 'REAL' 0 '0' 0]
[64 'CAUSE_POWER_GENERATION_TRANSMISSION_DISTRIBUTION' 'REAL' 0 '0' 0]
[65 'CAUSE_NATURAL' 'REAL' 0 '0' 0]
[66 'CAUSE_DEBRIS_AND_OPEN_BURNING' 'REAL' 0 '0' 0]
[67 'CAUSE_MISSING_DATA_NOT_SPECIFIED_UNDETERMINED' 'REAL' 0 '0' 0]
[68 'CAUSE_RECREATION_AND_CEREMONY' 'REAL' 0 '0' 0]
[69 'CAUSE_EQUIPMENT_AND_VEHICLE_USE' 'REAL' 0 '0' 0]
[70 'CAUSE_ARSON_INCENDIARISM' 'REAL' 0 '0' 0]
[71 'CAUSE_FIREWORKS' 'REAL' 0 '0' 0]
[72 'CAUSE_OTHER_CAUSES' 'REAL' 0 '0' 0]
[73 'CAUSE_RAILROAD_OPERATIONS_AND_MAINTENANCE' 'REAL' 0 '0' 0]
[74 'CAUSE_SMOKING' 'REAL' 0 '0' 0]
[75 'CAUSE_MISUSE_OF_FIRE_BY_A_MINOR' 'REAL' 0 '0' 0]
[76 'CAUSE_FIREARMS_AND_EXPLOSIVES_USE' 'REAL' 0 '0' 0]
[77 'REAL_DISCOVERY_DOY' 'REAL' 0 None 0]
[78 'REAL_FIRE_SIZE' 'REAL' 0 None 0]
[79 'REAL_LATITUDE' 'REAL' 0 None 0]
[80 'REAL_LONGITUDE' 'REAL' 0 None 0]

# Contributors

| Name | Username |
|---|---|
| Bilal Ayoub | bilalAhmadAyoub |
| Jethro Swanson | Jethro-Swanson |
| Kate Walley | KatieCodess |
| Henry Wong | hwbit |
