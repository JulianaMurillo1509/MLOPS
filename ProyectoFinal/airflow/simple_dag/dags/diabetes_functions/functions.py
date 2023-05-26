import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pycaret.regression import *
from sqlalchemy import create_engine, MetaData, Column, Integer, String, Float, Table, text
from sqlalchemy.orm import sessionmaker

DB_NAME= "postgres"
DB_USER="airflow"
DB_PASSWORD="airflow"
DB_HOST="192.168.0.155"
DB_PORT="5432"
os.environ['DB_NAME']=DB_NAME
os.environ['DB_USER']=DB_USER
os.environ['DB_PASSWORD']=DB_PASSWORD
os.environ['DB_HOST']=DB_HOST
os.environ['DB_PORT']=DB_PORT
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.102.111:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'



def connect_database():
    print('***connect_database***')
    print("DB_HOST", DB_HOST)
    print("DB_PORT", DB_PORT)
    # Connect to the database
    print('DB_PASSWORD',DB_PASSWORD)
    #engine = create_engine('postgresql://airflow:'+DB_PASSWORD+'@'+DB_HOST+':'+DB_PORT+'/airflow')
    engine = create_engine('postgresql+psycopg2://airflow:'+DB_PASSWORD+'@'+DB_HOST+'/postgres')
    # postgresql+psycopg2://airflow:***@postgres/airflow
    Session = sessionmaker(bind=engine)
    session = Session()
    print("session",session)
    print("engine",engine)
    return session,engine


def clean_data(df):
    Rep = df.replace('?', np.NaN)
    nacheck = Rep.isnull().sum()
    nacheck
    datacopy = df.drop(['id', 'weight', 'payer_code', 'medical_specialty'], axis=1)
    datacopy['30readmit'] = np.where(datacopy['readmitted'] == 'NO', 0, 1)
    datacopy = datacopy[((datacopy.discharge_disposition_id != 11) &
                         (datacopy.discharge_disposition_id != 13) &
                         (datacopy.discharge_disposition_id != 14) &
                         (datacopy.discharge_disposition_id != 19) &
                         (datacopy.discharge_disposition_id != 20) &
                         (datacopy.discharge_disposition_id != 21))]
    # Cleaning the data, replacing the null values in numeric data by 0 and object data by unknown,

    numcolumn = datacopy.select_dtypes(include=[np.number]).columns
    objcolumn = datacopy.select_dtypes(include=['object']).columns

    # Substituting 0 and unknown,

    datacopy[numcolumn] = datacopy[numcolumn].fillna(0)
    datacopy[objcolumn] = datacopy[objcolumn].fillna("unknown")

    # Encoding the data,

    def map_now():
        listname = [('infections', 139),
                    ('neoplasms', (239 - 139)),
                    ('endocrine', (279 - 239)),
                    ('blood', (289 - 279)),
                    ('mental', (319 - 289)),
                    ('nervous', (359 - 319)),
                    ('sense', (389 - 359)),
                    ('circulatory', (459 - 389)),
                    ('respiratory', (519 - 459)),
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
                if ((num is None) | (num in ['unknown', '?']) | (pd.isnull(num))):
                    temp.append('unknown')
                elif (num.upper()[0] == 'V'):
                    temp.append('supplemental')
                elif (num.upper()[0] == 'E'):
                    temp.append('injury')
                else:
                    lkup = num.split('.')[0]
                    temp.append(codes[lkup])
            df.loc[:, col] = temp
        return df

    listcol = ['diag_1', 'diag_2', 'diag_3']
    codes = map_now()
    datacopy[listcol] = codemap(datacopy[listcol], codes)

    data1 = datacopy.drop(['encounter_id', "patient_nbr", 'admission_type_id', 'readmitted'], axis=1)
    # Normalization of the data,

    listnormal = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                  'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    normal = StandardScaler()

    data1[listnormal] = normal.fit_transform(data1[listnormal])
    return data1


def read_data(data):
    print('***read_data***', data)
    session, engine = connect_database()
    print('***session***', session)
    # Execute a SELECT query on the diabetes table
    query = text("SELECT * FROM Diabetes")
    result = session.execute(query)
    # Create a pandas DataFrame from the query result
    diabetes = pd.DataFrame(result.fetchall(), columns=result.keys())
    print('***diabetes data:***', diabetes.head())
    print(diabetes)
    session.close()
    return diabetes