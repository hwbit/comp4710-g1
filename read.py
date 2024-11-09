import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(f"sqlite:///Data/FPA_FOD_20221014.sqlite")


with engine.connect() as connection:
    result = connection.execute(text(
        "SELECT LATITUDE, LONGITUDE FROM Fires WHERE NWCG_CAUSE_CLASSIFICATION = 'Human' LIMIT 20"
        )
    )
    for row in result:
        print(row)
