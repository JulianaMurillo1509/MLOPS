import os
from fastapi import APIRouter, FastAPI, HTTPException, File, UploadFile
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, MetaData, Column, Integer, String, Float, Table, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.declarative import declarative_base

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

@router.delete("/delete_All_data")
def delete_data():
    print('***delete_data***')
    session,engine=connect_database()
    # define the table to drop
    # Connect to the database and get a table object
    with engine.connect() as conn:
     try:
        # Get the table object
        metadata = MetaData()
        table = Table('covertype', metadata, autoload_with=engine)
        # Drop the table
        table.drop(bind=conn)
     except NoSuchTableError as  e:
            raise HTTPException(status_code=500, detail= 'table does not exist!' )  
        
    session.close()
        
    return "data deletion successful"


def insert_data(covertype):
    print('***insert_data***')
    # Connect to the database
    session, engine = connect_database()
    print("***df covertype***", covertype.info())
    # Define the table schema
    Base = declarative_base()
    # Define covertype table model
    class Covertype(Base):
        __tablename__ = 'covertype'
        id = Column(Integer, primary_key=True, index=True)
        Elevation= Column(Integer)
        Aspect = Column(Integer)
        Slope = Column(Integer)
        Horizontal_Distance_To_Hydrology = Column(Integer)
        Vertical_Distance_To_Hydrology = Column(Integer)
        Horizontal_Distance_To_Roadways = Column(Integer)
        Hillshade_9am = Column(Integer)
        Hillshade_Noon = Column(Integer)
        Hillshade_3pm=Column(Integer)
        Horizontal_Distance_To_Fire_Points = Column(Integer)
        Wilderness_Area=Column(String)
        Soil_Type = Column(String)
        Cover_Type = Column(Integer)

    print('Base',Base)
    # Create table if it doesn't exist
    Base.metadata.create_all(bind=engine)
    # Insert the data into the table
    print('Insert the data into the table')
    covertype_table = Table('covertype',Base.metadata, autoload=True)
    # Print schema of the covertype table
    print("covertype_table:",covertype_table)
    # Create a connection object
    print("*** example query results:",session.execute(text('SELECT * FROM covertype')))

    for i, row in covertype.iterrows():
        print("***i:",i)
        covertype = Covertype(    Elevation= row['Elevation'],
                                Aspect = row['Aspect'],
                                Slope = row['Slope'],
                                Horizontal_Distance_To_Hydrology = row['Horizontal_Distance_To_Hydrology'],
                                Vertical_Distance_To_Hydrology = row['Vertical_Distance_To_Hydrology'],
                                Horizontal_Distance_To_Roadways = row['Horizontal_Distance_To_Roadways'],
                                Hillshade_9am = row['Hillshade_9am'],
                                Hillshade_Noon = row['Hillshade_Noon'],
                                Hillshade_3pm=row['Hillshade_3pm'],
                                Horizontal_Distance_To_Fire_Points = row['Horizontal_Distance_To_Fire_Points'],
                                Wilderness_Area= row['Wilderness_Area'],
                                Soil_Type = row['Soil_Type'],
                                Cover_Type = row['Cover_Type'])
        session.add(covertype)

    print("***session before commit***",session)
    session.commit()
    print("*** example query results:", session.execute(text('SELECT * FROM covertype')))
    session.close()


def clean_data(df):
    # Display the number of missing values in each column
    print(df.isnull().sum())

    # Create a SimpleImputer object with strategy='mean'
    imputer = SimpleImputer(strategy='mean')

    # Fit the imputer to the dataset
    colums = ['bill_length_mm','bill_depth_mm','body_mass_g', 'flipper_length_mm']
    imputer.fit(df[colums])

    # Transform the dataset by filling the missing values with the mean
    df[colums] = imputer.transform(df[colums])
    return df

def read_data(data):
    print('***read_data***',data)
    session, engine = connect_database()
    print('***session***',session)
    # Execute a SELECT query on the covertype table
    query = text("SELECT * FROM covertype")
    result = session.execute(query)
    # Create a pandas DataFrame from the query result
    covertype = pd.DataFrame(result.fetchall(), columns=result.keys())
    print('***covertype data:***', covertype.head())
    print(covertype)
    session.close()
    return  covertype


@app.get("/")
async def root():
    return {"message": "Hello  training covertype"}

@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
        csv = pd.read_csv(file.file)
        insert_data(csv)
        return("finish insert succesfully")


@router.get("/train")
async def train_model(data:str='covertype'):
    print("***train_model***")
    df = read_data(data) #read data from data base
    print('***df:***',df.head())

    #cast pandas object to a specified dtype
    df["Wilderness_Area"] = df["Wilderness_Area"].astype('category')
    df["Soil_Type"] = df["Soil_Type"].astype('category')

    #. Return Series of codes as well as the index
    df["Wilderness_Area"] = df["Wilderness_Area"].cat.codes
    df["Soil_Type"] = df["Soil_Type"].cat.codes

    df = df.drop(['id'],axis = 1)

    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    path="/work1/"
    dump(model, path+data+'_model.joblib')
    print("model trained and safe in :",path+data+'_model.joblib')
    return accuracy




