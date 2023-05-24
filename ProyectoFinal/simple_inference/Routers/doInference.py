from Schemas.schemas import diabetes
import os
from fastapi import APIRouter, FastAPI, HTTPException
import numpy as np
import pandas as pd
from joblib import load
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

app = FastAPI()
router = APIRouter(
    prefix="/do_inference",
    tags= ['do inference']
)

@app.get("/")
async def root():
    return {"message": "Hello World inference"}


def prepare_sample(diabetes: diabetes): #prepare data before prediction
        _wilderness_area_map = {'Cache': 2, 'Commanche': 0, 'Rawah': 1,'Neota': 3,}
        _Soil_type_map = {'C2702':0,'C2703':1,'C2704':2,'C2705':3,'C2706':4,'C2717':5,'C3501':6,'C3502':7,'C4201':8,'C4703':9,'C4704':10,
                          'C4744':11,'C4758':12,'C5101':13,'C5151':14,'C6101':15,'C6102':16,'C6731':17,'C7101':18,'C7102':19,'C7103':20,
                          'C7201':21,'C7202':22,'C7700':23,'C7701':24,'C7702':25,'C7709':26,'C7710':27,'C7745':28,'C7746':29,
                          'C7755':30,'C7756':31,'C7757':32,'C7790':33,'C8703':34,'C8707':35,'C8708':36,'C8771':37,'C872':38, 'C876':39}

        wilderness = _wilderness_area_map[diabetes.Wilderness_Area]
        soil = _Soil_type_map[diabetes.Soil_Type]
        sample = [diabetes.Elevation,diabetes.Aspect,diabetes.Slope,diabetes.Horizontal_Distance_To_Hydrology,
                  diabetes.Vertical_Distance_To_Hydrology,diabetes.Horizontal_Distance_To_Roadways,diabetes.Hillshade_9am,
                  diabetes.Hillshade_Noon,diabetes.Hillshade_3pm,diabetes.Horizontal_Distance_To_Fire_Points,
                  wilderness, soil]
        sample = np.array(sample).reshape(1, -1)

        return sample

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


def insert_data(diabetes):
    print('***insert_data***')
    # Connect to the database
    session, engine = connect_database()
    print("session",session)
    print("engine", engine)
    print("***df diabetes***", diabetes.info())
    # Define the table schema
    Base = declarative_base()
    # Define diabetes table model

    class Diabetes(Base):
        __tablename__ = 'diabetes'      
        id = Column(Integer, primary_key=True)
        encounter_id = Column(Integer)
        patient_nbr = Column(Integer)
        race = Column(String)
        gender = Column(String)
        age = Column(String)
        weight = Column(String)
        admission_type_id = Column(Integer)
        discharge_disposition_id = Column(Integer)
        admission_source_id = Column(Integer)
        time_in_hospital = Column(Integer)
        payer_code = Column(String)
        medical_specialty = Column(String)
        num_lab_procedures = Column(Integer)
        num_procedures = Column(Integer)
        num_medications = Column(Integer)
        number_outpatient = Column(Integer)
        number_emergency = Column(Integer)
        number_inpatient = Column(Integer)
        diag_1 = Column(String)
        diag_2 = Column(String)
        diag_3 = Column(String)
        number_diagnoses = Column(Integer)
        max_glu_serum = Column(String)
        A1Cresult = Column(String)
        metformin = Column(String)
        repaglinide = Column(String)
        nateglinide = Column(String)
        chlorpropamide = Column(String)
        glimepiride = Column(String)
        acetohexamide = Column(String)
        glipizide = Column(String)
        glyburide = Column(String)
        tolbutamide = Column(String)
        pioglitazone = Column(String)
        rosiglitazone = Column(String)
        acarbose = Column(String)
        miglitol = Column(String)
        troglitazone = Column(String)
        tolazamide = Column(String)
        examide = Column(String)
        citoglipton = Column(String)
        insulin = Column(String)
        glyburide_metformin = Column(String)
        glipizide_metformin = Column(String)
        glimepiride_pioglitazone = Column(String)
        metformin_rosiglitazone = Column(String)
        metformin_pioglitazone = Column(String)
        change = Column(String)
        diabetesMed = Column(String)
        readmitted = Column(String)

    print('Base',Base)
    # Create table if it doesn't exist
    Base.metadata.create_all(bind=engine)
    # Insert the data into the table
    print('Insert the data into the table')
    diabetes_table = Table('diabetes',Base.metadata, autoload=True)
    # Print schema of the diabetes table
    print("diabetes_table:",diabetes_table)
    # Create a connection object
    print("*** example query results:",session.execute(text('SELECT * FROM diabetes order by id desc limit 10')))

    for i, row in diabetes.iterrows():
     try:
       def insert_data(diabetes):
        print('***insert_data***')
        # Connect to the database
        session, engine = connect_database()
        print("session",session)
        print("engine", engine)
        print("***df diabetes***", diabetes.info())
        # Define the table schema
        Base = declarative_base()
        # Define diabetes table model

        class Diabetes(Base):
            __tablename__ = 'diabetes'      
            id = Column(Integer, primary_key=True)
            encounter_id = Column(Integer)
            patient_nbr = Column(Integer)
            race = Column(String)
            gender = Column(String)
            age = Column(String)
            weight = Column(String)
            admission_type_id = Column(Integer)
            discharge_disposition_id = Column(Integer)
            admission_source_id = Column(Integer)
            time_in_hospital = Column(Integer)
            payer_code = Column(String)
            medical_specialty = Column(String)
            num_lab_procedures = Column(Integer)
            num_procedures = Column(Integer)
            num_medications = Column(Integer)
            number_outpatient = Column(Integer)
            number_emergency = Column(Integer)
            number_inpatient = Column(Integer)
            diag_1 = Column(String)
            diag_2 = Column(String)
            diag_3 = Column(String)
            number_diagnoses = Column(Integer)
            max_glu_serum = Column(String)
            A1Cresult = Column(String)
            metformin = Column(String)
            repaglinide = Column(String)
            nateglinide = Column(String)
            chlorpropamide = Column(String)
            glimepiride = Column(String)
            acetohexamide = Column(String)
            glipizide = Column(String)
            glyburide = Column(String)
            tolbutamide = Column(String)
            pioglitazone = Column(String)
            rosiglitazone = Column(String)
            acarbose = Column(String)
            miglitol = Column(String)
            troglitazone = Column(String)
            tolazamide = Column(String)
            examide = Column(String)
            citoglipton = Column(String)
            insulin = Column(String)
            glyburide_metformin = Column(String)
            glipizide_metformin = Column(String)
            glimepiride_pioglitazone = Column(String)
            metformin_rosiglitazone = Column(String)
            metformin_pioglitazone = Column(String)
            change = Column(String)
            diabetesMed = Column(String)
            readmitted = Column(String)

        print('Base',Base)
        # Create table if it doesn't exist
        Base.metadata.create_all(bind=engine)
        # Insert the data into the table
        print('Insert the data into the table')
        diabetes_table = Table('diabetes',Base.metadata, autoload=True)
        # Print schema of the diabetes table
        print("diabetes_table:",diabetes_table)
        # Create a connection object
        print("*** example query results:",session.execute(text('SELECT * FROM diabetes order by id desc limit 10')))

        for i, row in diabetes.iterrows():
            print("***i:",i)
            diabetes = Diabetes(    encounter_id=row['encounter_id'],
                                    patient_nbr=row['patient_nbr'],
                                    race=row['race'],
                                    gender=row['gender'],
                                    age=row['age'],
                                    weight=row['weight'],
                                    admission_type_id=row['admission_type_id'],
                                    discharge_disposition_id=row['discharge_disposition_id'],
                                    admission_source_id=row['admission_source_id'],
                                    time_in_hospital=row['time_in_hospital'],
                                    payer_code=row['payer_code'],
                                    medical_specialty=row['medical_specialty'],
                                    num_lab_procedures=row['num_lab_procedures'],
                                    num_procedures=row['num_procedures'],
                                    num_medications=row['num_medications'],
                                    number_outpatient=row['number_outpatient'],
                                    number_emergency=row['number_emergency'],
                                    number_inpatient=row['number_inpatient'],
                                    diag_1=row['diag_1'],
                                    diag_2=row['diag_2'],
                                    diag_3=row['diag_3'],
                                    number_diagnoses=row['number_diagnoses'],
                                    max_glu_serum=row['max_glu_serum'],
                                    A1Cresult=row['A1Cresult'],
                                    metformin=row['metformin'],
                                    repaglinide=row['repaglinide'],
                                    nateglinide=row['nateglinide'],
                                    chlorpropamide=row['chlorpropamide'],
                                    glimepiride=row['glimepiride'],
                                    acetohexamide=row['acetohexamide'],
                                    glipizide=row['glipizide'],
                                    glyburide=row['glyburide'],
                                    tolbutamide=row['tolbutamide'],
                                    pioglitazone=row['pioglitazone'],
                                    rosiglitazone=row['rosiglitazone'],
                                    acarbose=row['acarbose'],
                                    miglitol=row['miglitol'],
                                    troglitazone=row['troglitazone'],
                                    tolazamide=row['tolazamide'],
                                    examide=row['examide'],
                                    citoglipton=row['citoglipton'],
                                    insulin=row['insulin'],
                                    glyburide_metformin=row['glyburide-metformin'],
                                    glipizide_metformin=row['glipizide-metformin'],
                                    glimepiride_pioglitazone=row['glimepiride-pioglitazone'],
                                    metformin_rosiglitazone=row['metformin-rosiglitazone'],
                                    metformin_pioglitazone=row['metformin-pioglitazone'],
                                    change=row['change'],
                                    diabetesMed=row['diabetesMed'],
                                    readmitted=row['readmitted'])
            session.add(diabetes)


        print("***session before commit***",session)
        session.commit()
        # print("*** example query results:", session.execute(text('SELECT * FROM Diabetes')))
        session.close()


     except Exception as e:
        print("ERROR:",e)
        result=e
    return result


@router.post("/diabetes")
async def test_model(diabetes: diabetes, model:str='diabetes'):

    input_df = pd.DataFrame([diabetes.dict()])


    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.102.111:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    # crea un cliente MLflow
    client = MlflowClient()

    mlflow.set_tracking_uri("http://10.43.102.111:5000")

    # Obtiene la información de todas las versiones de la mas reciente a la más antigua
    all_versions = client.get_latest_versions("final_best_production", stages= ["Production"])
    latest_model_name = all_versions[-1].name
    latest_model_stage = all_versions[-1].current_stage
    latest_version = all_versions[-1].version
    model_name = "final_best_production"

    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
    model_loaded = mlflow.pyfunc.load_model(model_uri=model_production_uri)

    prediction = model_loaded.predict(input_df)


    return {f'el modelo que se utilizó para la inferencia es {latest_model_name} que esta en {latest_model_stage} su versión es {latest_version} y la predicción del diabetes es': int(prediction[0])}