import requests
import os
from fastapi import HTTPException, APIRouter
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.svm import SVC
from sklearn import metrics

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Table
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi import FastAPI
import pandas as pd

DB_PASSWORD=os.environ['DB_PASSWORD']

app = FastAPI()
router = APIRouter(
    prefix="/train_model",
    tags= ['train models']
)


def connect_database():
    print('***connect_database***')
    # Connect to the database
    print('DB_PASSWORD',DB_PASSWORD)
    engine = create_engine('postgresql://myuser:mypassword@db/mydatabase')
    Session = sessionmaker(bind=engine)
    session = Session()
    print("session",session)
    print("engine",engine)

    return session,engine


def delete_data():
    print('***delete_data***')
    session,engine=connect_database()
    # define the table to drop
    # Connect to the database and get a table object
    with engine.connect() as conn:
        # Get the table object
        table = Table('penguins', autoload=True, autoload_with=engine)
        # Drop the table
        table.drop()
    session.close()



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
        bill_length_mm=Column(Integer)
        bill_depth_mm = Column(Integer)
        flipper_length_mm=Column(Integer)
        body_mass_g = Column(Integer)

    print('Base',Base)
    # Create table if it doesn't exist
    Base.metadata.create_all(bind=engine)
    # Insert the data into the table
    print('Insert the data into the table')
    for i, row in penguins.iterrows():
        penguin = Penguin(species=row['species'],
                          island=row['island'],
                          bill_length_mm=row['bill_length_mm'],
                          bill_depth_mm=row['bill_depth_mm'],
                          flipper_length_mm=row['flipper_length_mm'],
                          body_mass_g=row['body_mass_g'])
        session.add(penguin)

    session.commit()
    session.close()


def get_data(data):
    print('***get_data***')
    if data=='penguins':
        penguins = pd.read_csv(
            'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv')
        print('penguins',penguins.head())
        insert_data(penguins)
        print("finish insert")
    else:
        raise HTTPException(status_code=500, detail="Unkown dataset: "+ data)


def read_data(data):
    print('***read_data***')
    session, engine = connect_database()
    penguins = session.query(data).all()
    for penguin in penguins[:-10]:
        print(penguin.species, penguin.island, penguin.bill_length_mm, penguin.bill_depth_mm, penguin.flipper_length_mm,
              penguin.body_mass_g, penguin.sex)
    session.close()
    return  penguins


@app.get("/")
async def root():
    return {"message": "Hello  training penguins"}


@router.get("/train")
async def train_model(data:str='penguins'):
    get_data(data) #get data from source and insert in db
    df = read_data(data)
    print('df',df.head())
    df.columns = df.columns.str.replace(' ', '_')
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = SVC()
    model.fit(X_train, y_train)
    expected_y  = y_test
    predicted_y = model.predict(X_test)
    model_metrics = metrics.classification_report(expected_y, predicted_y, output_dict=True,zero_division=1)
    dump(model, data+'_model.joblib')
    return model_metrics




