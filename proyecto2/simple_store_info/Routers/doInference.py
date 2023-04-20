from Schemas.schemas import Covertype
import os
from fastapi import APIRouter, FastAPI, HTTPException
import numpy as np
import pandas as pd
from joblib import load
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, MetaData, Column, Integer, String, Float, Table, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.declarative import declarative_base


DB_PASSWORD=os.environ['DB_PASSWORD']
DB_HOST=os.environ['DB_HOST']
DB_PORT=os.environ['DB_PORT']

app = FastAPI()
router = APIRouter(
    prefix="/store",
    tags= ['store info in csv']
)

@app.get("/")
async def root():
    return {"message": "Hello World inference"}


def connect_database():
    print('***connect_database***')
    print("DB_HOST", DB_HOST)
    print("DB_PORT", DB_PORT)
    # Connect to the database
    print('DB_PASSWORD',DB_PASSWORD)
    engine = create_engine('postgresql://myuser:'+DB_PASSWORD+'@'+DB_HOST+':'+DB_PORT+'/mydatabase')
    Session = sessionmaker(bind=engine)
    session = Session()
    print("session",session)
    print("engine",engine)
    return session,engine


@app.get("/convert_inference_csv")
async def read_data():
    # Connect to the PostgreSQL database
    session,engine=connect_database()
    #db_url = "postgresql://user:password@db:5432/database"
    #engine = create_engine(db_url)

    # Define the table metadata
    metadata = MetaData()
    covertype = Table('covertype_inference', metadata, autoload=True, autoload_with=engine)

    # Load the data into a Pandas dataframe
    df = pd.read_sql_table('covertype_inference', engine)

    # Create the CSV file
    csv_file = 'covertype_inference.csv'
    df.to_csv(csv_file, index=False)

    # Create the directory for the shared volume if it doesn't exist
    if not os.path.exists('/inference'):
        os.makedirs('/inference')

    # Write the CSV file to the shared volume
    with open('/inference/{}'.format(csv_file), 'w') as f:
        f.write(df.to_csv(index=False))

    return {'message': 'Data loaded successfully and CSV written to shared volume'}


