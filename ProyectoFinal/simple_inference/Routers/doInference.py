from Schemas.schemas import Diabetes
import json
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


@router.post("/diabetes")
async def test_model(diabetes: Diabetes, model:str='diabetes'):
    diabetes_dict = {
        'race': diabetes.race,
        'gender': diabetes.gender,
        'age': diabetes.age,
        'discharge_disposition_id': diabetes.discharge_disposition_id,
        'admission_source_id': diabetes.admission_source_id,
        'time_in_hospital': diabetes.time_in_hospital,
        'num_lab_procedures': diabetes.num_lab_procedures,
        'num_procedures': diabetes.num_procedures,
        'num_medications': diabetes.num_medications,
        'number_outpatient': diabetes.number_outpatient,
        'number_emergency': diabetes.number_emergency,
        'number_inpatient': diabetes.number_inpatient,
        'diag_1': diabetes.diag_1,
        'diag_2': diabetes.diag_2,
        'diag_3': diabetes.diag_3,
        'number_diagnoses': diabetes.number_diagnoses,
        'max_glu_serum': diabetes.max_glu_serum,
        'A1Cresult': diabetes.A1Cresult,
        'metformin': diabetes.metformin,
        'repaglinide': diabetes.repaglinide,
        'nateglinide': diabetes.nateglinide,
        'chlorpropamide': diabetes.chlorpropamide,
        'glimepiride': diabetes.glimepiride,
        'acetohexamide': diabetes.acetohexamide,
        'glipizide': diabetes.glipizide,
        'glyburide': diabetes.glyburide,
        'tolbutamide': diabetes.tolbutamide,
        'pioglitazone': diabetes.pioglitazone,
        'rosiglitazone': diabetes.rosiglitazone,
        'acarbose': diabetes.acarbose,
        'miglitol': diabetes.miglitol,
        'troglitazone': diabetes.troglitazone,
        'tolazamide': diabetes.tolazamide,
        'examide': diabetes.examide,
        'citoglipton': diabetes.citoglipton,
        'insulin': diabetes.insulin,
        'glyburide_metformin': diabetes.glyburide_metformin,
        'glipizide_metformin': diabetes.glipizide_metformin,
        'glimepiride_pioglitazone': diabetes.glimepiride_pioglitazone,
        'metformin_rosiglitazone': diabetes.metformin_rosiglitazone,
        'metformin_pioglitazone': diabetes.metformin_pioglitazone,
        'change': diabetes.change,
        'diabetesMed': diabetes.diabetesMed
    }
    diabetes_df = pd.DataFrame([diabetes_dict])
    column_order = ['race', 'gender', 'age', 'discharge_disposition_id',
                'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
                'num_procedures', 'num_medications', 'number_outpatient',
                'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
                'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',
                'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                'tolazamide', 'examide', 'citoglipton', 'insulin',
                'glyburide_metformin', 'glipizide_metformin',
                'glimepiride_pioglitazone', 'metformin_rosiglitazone',
                'metformin_pioglitazone', 'change', 'diabetesMed']
    
    input_df = diabetes_df[column_order]

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.102.111:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    # crea un cliente MLflow
    client = MlflowClient()
    mlflow.set_tracking_uri("http://10.43.102.111:5000")

    model_name = "final_best_production_Last_Project_MLOPS"
    # Obtiene la información de todas las versiones de la mas reciente a la más antigua
    all_versions = client.get_latest_versions("final_best_production_Last_Project_MLOPS", stages= ["Production"])
    if len(all_versions) > 1:
        old_version = all_versions[0].version

        # Transition the model to production
        client.transition_model_version_stage(
            name=model_name,
            version=old_version,
            stage="Archived"
        )

    latest_model_name = all_versions[-1].name
    latest_model_stage = all_versions[-1].current_stage
    latest_version = all_versions[-1].version
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
    model_loaded = mlflow.pyfunc.load_model(model_uri=model_production_uri)
    prediction = model_loaded.predict(input_df)


    return {f'el modelo que se utilizó para la inferencia es {latest_model_name} que esta en {latest_model_stage} su versión es {latest_version} y la predicción es': int(prediction[0])}