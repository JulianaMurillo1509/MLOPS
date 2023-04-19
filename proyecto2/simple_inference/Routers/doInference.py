from Schemas.schemas import Penguin
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
    prefix="/do_inference",
    tags= ['do inference']
)

@app.get("/")
async def root():
    return {"message": "Hello World inference"}


def prepare_sample(penguin: Penguin): #prepare data before prediction
        _island_map = {'Torgersen': 2, 'Biscoe': 0, 'Dream': 1}
        _sex_map = {'male': 2, 'female': 1}

        island = _island_map[penguin.island]
        sex = _sex_map[penguin.sex]
        sample = [penguin.bill_length_mm, penguin.bill_depth_mm, penguin.flipper_length_mm, penguin.body_mass_g, island, sex,penguin.year]
        sample = np.array([np.asarray(sample)]).reshape(-1, 1)

        return sample.reshape(1, -1)

def connect_database():
    print('***connect_database***')
    print("DB_HOST", DB_HOST)
    print("DB_PORT", DB_PORT)
    # Connect to the database
    print('DB_PASSWORD',DB_PASSWORD)
    engine = create_engine('postgresql://myuser:mypassword@DB_HOST:DB_PORT/mydatabase')
    Session = sessionmaker(bind=engine)
    session = Session()
    print("session",session)
    print("engine",engine)
    return session,engine


def insert_data(penguins):
    print('***insert_data***')
    # Connect to the database
    session, engine = connect_database()
    print("***df penguins***", penguins.info())
    # Define the table schema
    Base = declarative_base()
    # Define penguins table model
    class Penguin(Base):
        __tablename__ = 'penguins'
        id = Column(Integer, primary_key=True, index=True)
        species = Column(String)
        island = Column(String)
        bill_length_mm=Column(Float)
        bill_depth_mm = Column(Float)
        flipper_length_mm=Column(Float)
        body_mass_g = Column(Float)
        sex = Column(String)
        year = Column(Integer)

    print('Base',Base)
    # Create table if it doesn't exist
    Base.metadata.create_all(bind=engine)
    # Insert the data into the table
    print('Insert the data into the table')
    penguins_table = Table('penguins',Base.metadata, autoload=True)
    # Print schema of the penguins table
    print("penguins_table:",penguins_table)
    # Create a connection object
    print("*** example query results:",session.execute(text('SELECT * FROM penguins order by id desc limit 10' )))

    for i, row in penguins.iterrows():
        print("***i:",i)
        #species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year
        penguin = Penguin(species=row['species'],
                          island=row['island'],
                          bill_length_mm=row['bill_length_mm'],
                          bill_depth_mm=row['bill_depth_mm'],
                          flipper_length_mm=row['flipper_length_mm'],
                          body_mass_g=row['body_mass_g'],
                          sex = row['sex'],
                          year = row['year'])
        session.add(penguin)
    print("***session before commit***",session)
    session.commit()
    print("*** example query results:", session.execute(text('SELECT * FROM penguins')))
    session.close()


@router.post("/penguins")
async def test_model(penguin: Penguin, model:str='penguins'):
    path = "/work2/"
    print("path:",path+model+'_model.joblib')
    if not os.path.isfile(path+model+'_model.joblib'):
        raise HTTPException(status_code=500, detail="Unkown model: "+ model+" Try to train model first.")
    print("loadign mdoel from:",path+model+'_model.joblib')
 
    spicies_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    preparedData = prepare_sample(penguin)
    model_loaded = load(path+model+'_model.joblib')
    species = model_loaded.predict(preparedData)[0]
    result = (spicies_map[species])
    answer = "the penguin specie is:" + result
    return answer



