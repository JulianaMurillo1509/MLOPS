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
    #engine = create_engine('postgresql://myuser:' + DB_PASSWORD + '@' + DB_HOST + '/mydatabase')
    #engine = create_engine('postgresql://airflow:' + DB_PASSWORD + '@' + DB_HOST + '/postgres')
    # postgresql+psycopg2://airflow:***@postgres/airflow
    engine = create_engine('postgresql+psycopg2://airflow:airflow@10.43.102.111:5432/postgres')
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
        table = Table('diabetes', metadata, autoload_with=engine)
        # Drop the table
        table.drop(bind=conn)
     except NoSuchTableError as  e:
            raise HTTPException(status_code=500, detail= 'table does not exist!' )  
        
    session.close()
        
    return "data deletion successful"


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



def clean_data(df):

    Rep = df.replace('?', np.NaN) 
    nacheck = Rep.isnull().sum()
    datacopy= df.drop(['id','weight','payer_code','medical_specialty'],axis=1)
    datacopy['readmit'] = np.where(datacopy['readmitted'] == 'NO', 0, 1)
    datacopy = datacopy[((datacopy.discharge_disposition_id != 11) & 
                                          (datacopy.discharge_disposition_id != 13) &
                                          (datacopy.discharge_disposition_id != 14) & 
                                          (datacopy.discharge_disposition_id != 19) & 
                                          (datacopy.discharge_disposition_id != 20) & 
                                          (datacopy.discharge_disposition_id != 21))] 
    # Cleaning the data, replacing the null values in numeric data by 0 and object data by unknown,

    numcolumn = datacopy.select_dtypes(include = [np.number]).columns
    objcolumn = datacopy.select_dtypes(include = ['object']).columns

    # Substituting 0 and unknown,

    datacopy[numcolumn] = datacopy[numcolumn].fillna(0)
    datacopy[objcolumn] = datacopy[objcolumn].fillna("unknown")

    #Encoding the data,

    def map_now():
        listname = [('infections', 139),
                    ('neoplasms', (239 - 139)),
                    ('endocrine', (279 - 239)),
                    ('blood', (289 - 279)),
                    ('mental', (319 - 289)),
                    ('nervous', (359 - 319)),
                    ('sense', (389 - 359)),
                    ('circulatory', (459-389)),
                    ('respiratory', (519-459)),
                    ('digestive', (579 - 519)),
                    ('genitourinary', (629 - 579)),
                    ('pregnancy', (679 - 629)),
                    ('skin', (709 - 679)),
                    ('musculoskeletal', (739 - 709)),
                    ('congenital', (759 - 739)),
                    ('perinatal', (779 - 759)),
                    ('ill-defined', (799 - 779)),
                    ('injury', (999 - 799))]
        
        
        dictcout = {}
        count = 1
        for name, num in listname:
            for i in range(num):
                dictcout.update({str(count): name})  
                count += 1
        return dictcout
    

    def codemap(df, codes):
        import pandas as pd
        namecol = df.columns.tolist()
        for col in namecol:
            temp = [] 
            for num in df[col]:           
                if ((num is None) | (num in ['unknown', '?']) | (pd.isnull(num))): temp.append('unknown')
                elif(num.upper()[0] == 'V'): temp.append('supplemental')
                elif(num.upper()[0] == 'E'): temp.append('injury')
                else: 
                    lkup = num.split('.')[0]
                    temp.append(codes[lkup])           
            df.loc[:, col] = temp               
        return df 


    listcol = ['diag_1', 'diag_2', 'diag_3']
    codes = map_now()
    datacopy[listcol] = codemap(datacopy[listcol], codes)

    data1 = datacopy.drop(['encounter_id', "patient_nbr", 'admission_type_id','readmitted'], axis =1) 
    #Normalization of the data,

    listnormal = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                     'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    normal = StandardScaler()

    data1[listnormal] = normal.fit_transform(data1[listnormal])
    return data1
    

def read_data(data):
    print('***read_data***',data)
    session, engine = connect_database()
    print('***session***',session)
    # Execute a SELECT query on the diabetes table
    query = text("SELECT * FROM Diabetes")
    result = session.execute(query)
    # Create a pandas DataFrame from the query result
    diabetes = pd.DataFrame(result.fetchall(), columns=result.keys())
    print('***diabetes data:***', diabetes.head())
    print(diabetes)
    session.close()
    return  diabetes


@app.get("/")
async def root():
    return {"message": "Hello  training diabetes"}

@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
        csv = pd.read_csv(file.file)
        insert_data(csv)
        return("finish insert succesfully")


@router.get("/train")
async def train_model(data:str='diabetes_clean'):
    print("***train_model***")
    df = read_data(data) #read data from data base
    clean_df = clean_data(df)

    #Let's store readmitted in y and rest of the columns in X,

    Y = clean_df['readmit']
    X = clean_df.drop(['readmit'], axis =1)
    X = pd.get_dummies(X)

    #Splitting the data into training and vallidation data sets. The training data will contain 80 % of the data and validation will contain remaining 20%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, 
                                                random_state = 7, stratify = Y)
    
    print("shape of Xtrain,Xtest:",X_train.shape,X_test.shape)

    
    # Connect to MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # Enable autologging in MLflow
    mlflow.autolog(log_model_signatures=True, log_input_examples=True)

    # Set up the data
    s = setup(clean_df, target = 'readmit', transform_target = True, log_experiment = True, experiment_name = 'Experimento proyecto final MLOPS 2023')
    # Compare models
    best_models = compare_models(n_select=10, sort='R2', fold=5)

    # compare models
    best = compare_models()
    # finalize the model
    final_best = finalize_model(best)
    # save model to disk
    save_model(final_best, 'final_best_production_Last_Project_MLOPS')

    # Register the model in MLflow
    mlflow.sklearn.log_model(final_best, 'final_best', registered_model_name='final_best_production_Last_Project_MLOPS')


    # Add the best model to production
    model_name = "final_best_production_Last_Project_MLOPS"

    # Create an MLflow client
    client = MlflowClient()

    # Get information about all versions without stage, from the most recent to the oldest
    all_versions = client.get_latest_versions(model_name, stages=None)

    print('Versions:', all_versions)

    # Check if there are registered versions of the model
    if len(all_versions) > 0:
        # Get the latest registered version of the model
        latest_version = all_versions[-1].version
        print(f"The latest version of the model is version {latest_version}.")
    else:
        print("There are no registered versions of the model.")

    # Transition the model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )

    # Get the feature names used during training
    training_columns = s.data.columns[:-1]  # Exclude the target column

    # Check the columns in your test dataset (X_test)
    print(X_test.columns)

    # Drop columns not present in the training data
    columns_to_drop = set(X_test.columns) - set(training_columns)
    X_test.drop(columns_to_drop, axis=1, inplace=True)

    # Add missing columns with default values
    columns_to_add = set(training_columns) - set(X_test.columns)
    for column in columns_to_add:
        X_test[column] = 'Unknown'  # Assign a default value to the new column

    # Ensure the order of columns matches the training data
    X_test = X_test[training_columns]

    # Predict using the aligned test dataset
    y_pred = final_best.predict(X_test)

    # Evaluate the best model on the test set
    
    y_pred = final_best.predict(X_test)
    metricR2 = metrics.r2_score(Y_test, y_pred)
    print("R2 Score:", metricR2)
    return f"This model was trained with version {latest_version} and has an R2 score of {metricR2}"