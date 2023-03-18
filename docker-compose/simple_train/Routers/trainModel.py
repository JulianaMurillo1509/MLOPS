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
from sqlalchemy import create_engine, Table,text
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi import FastAPI
import pandas as pd
from sklearn.impute import SimpleImputer

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
    print("*** example query results:",session.execute(text('SELECT * FROM penguins')))

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


def clean_data(df):
    # Display the number of missing values in each column
    print(df.isnull().sum())

    # Create a SimpleImputer object with strategy='mean'
    imputer = SimpleImputer(strategy='mean')

    # Fit the imputer to the dataset
    imputer.fit(df[['body_mass_g', 'flipper_length_mm']])

    # Transform the dataset by filling the missing values with the mean
    df[['body_mass_g', 'flipper_length_mm']] = imputer.transform(df[['body_mass_g', 'flipper_length_mm']])
    return df

def get_data(data):
    print('***get_data***')
    if data=='penguins':
        penguins = pd.read_csv(
            'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv')
        #species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year
        print('penguins',penguins.head())
        penguins=clean_data(penguins) #clean data
        insert_data(penguins)
        print("finish insert")
    else:
        raise HTTPException(status_code=500, detail="Unkown dataset: "+ data)


def read_data(data):
    print('***read_data***',data)
    session, engine = connect_database()
    print('***session***',session)
    # Execute a SELECT query on the penguins table
    query = text("SELECT * FROM penguins")
    result = session.execute(query)
    # Create a pandas DataFrame from the query result
    penguins = pd.DataFrame(result.fetchall(), columns=result.keys())
    print('***penguins data:***', penguins.head())
    for penguin in penguins[:-2]:
        print(penguin.species, penguin.island, penguin.bill_length_mm, penguin.bill_depth_mm, penguin.flipper_length_mm,
              penguin.body_mass_g, penguin.sex,penguin.year)
    session.close()
    return  penguins


@app.get("/")
async def root():
    return {"message": "Hello  training penguins"}


@router.get("/train")
async def train_model(data:str='penguins'):
    print("***train_model***")
    get_data(data) #get data from source and insert in db
    df = read_data(data) #read data from data base
    print('***df:***',df.head())
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




