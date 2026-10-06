import pandas as pd
import sqlite3
from pathlib import Path

def load_rides():
    path=Path(__file__).resolve().parent.parent/'data'/'rides.csv'
    return pd.read_csv(path)

def build_db(df):
    con=sqlite3.connect(':memory:',check_same_thread=False)
    df.to_sql('rides',con,index=False,if_exists='replace')
    return con

def execute_sql(con,sql):
    return pd.read_sql_query(sql,con)

def ride_fare_estimate(distance_km,vehicle_type='Ola Mini'):
    base={'Ola Auto':35,'Ola Mini':60,'Ola Prime':100,'Ola Prime SUV':125}.get(vehicle_type,60)
    per_km={'Ola Auto':14,'Ola Mini':18,'Ola Prime':28,'Ola Prime SUV':33}.get(vehicle_type,18)
    return round(base+distance_km*per_km,2)
