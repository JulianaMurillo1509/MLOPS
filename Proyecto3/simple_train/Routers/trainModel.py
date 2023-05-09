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
from pycaret.regression import *
import mlflow
from mlflow.tracking import MlflowClient

from sqlalchemy import create_engine, MetaData, Column, Integer, String, Float, Table, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.declarative import declarative_base

DB_PASSWORD=os.environ['DB_PASSWORD']
DB_HOST=os.environ['DB_HOST']
DB_PORT=os.environ['DB_PORT']

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.102.111:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'



app = FastAPI()
router = APIRouter(
    prefix="/train_model",
    tags= ['train models']
)

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
    print("session",session)
    print("engine", engine)
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
    print("*** example query results:",session.execute(text('SELECT * FROM covertype order by id desc limit 10')))

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
    # print("*** example query results:", session.execute(text('SELECT * FROM covertype')))
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

@router.post("/insertdataFromUrl")
async def get_data(data:str='penguins'):
    try:
        print("get_data:",data)
        get_data(data)
    except Exception as e:
        print("ERROR",e)
        return e
    return ("data inserted from url OK")


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

    
    # connect to mlflow
    mlflow.set_tracking_uri("http://10.43.102.111:5000")
    mlflow.set_experiment("test 1")

    mlflow.autolog(log_model_signatures=True, log_input_examples=True)

    s = setup(df, target = 'Cover_Type', transform_target = True, log_experiment = True, experiment_name = 'test 1')
    best = compare_models()
    # finalize the model
    final_best = finalize_model(best)
    best_model_name = best[0].__class__.__name__
    print(best_model_name)
    # save model to disk
    save_model(final_best, 'test1')
   
    # Registra el modelo en MLflow
    mlflow.sklearn.log_model(final_best, 'final_best', registered_model_name='final_best_production')
    #agrega el modelo a producción
    model_name = "final_best_production"

    # crea un cliente MLflow
    client = MlflowClient()

    # Obtiene la información de todas las versiones de la mas reciente a la más antigua
    all_versions = client.get_latest_versions("final_best_production", stages=None)
    print('versiones',all_versions)
    # Verifica si hay versiones registradas del modelo
    if len(all_versions) > 0:
        # Obtiene la última versión registrada del modelo
        latest_version = all_versions[-1].version
        print(f"La última versión del modelo es la versión {latest_version}.")
    else:
        print("No hay versiones registradas del modelo.")

    # Cambia el modelo a produccion
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )

    model = final_best
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metric = metrics.r2_score(y_test,y_pred)

    return f"Este modelo se entreno con la version {latest_version} de {best_model_name} y su r2_score es {metric}"


@router.get("/train_from_csv")
async def train_from_csv(data:str='covertype'):
    print("csv_to_df",data)
    # Path to the directory containing the CSV files
    csv_dir = "/inference/"

    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    print("csv_files", csv_files)
    # Create an empty list to hold the dataframes
    dfs = []

    # Loop through each CSV file
    for csv_file in csv_files:
        print("csv_file",csv_file)
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(os.path.join(csv_dir, csv_file))
        df.drop('id', axis=1, inplace=True)
        # Append the DataFrame to the list
        dfs.append(df)
    print("dfs", dfs)
    # Concatenate all the DataFrames together
    combined_df = pd.concat(dfs)
    print("combined_df",combined_df)
    print("insert combined_df",insert_data(combined_df))
    r=train_model(data)
    return r


