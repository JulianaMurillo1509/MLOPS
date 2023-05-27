import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pycaret.regression import *
from sqlalchemy import create_engine, MetaData, Column, Integer, String, Float, Table, text
from sqlalchemy.orm import sessionmaker
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

DB_NAME= "postgres"
DB_USER="airflow"
DB_PASSWORD="airflow"
DB_HOST="192.168.3.157"
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


def clean_data(data):
    print('***clean_data***', data)
    #leer data de diabetes_clean
    df=read_data("Diabetes")
    Rep = df.replace('?', np.NaN)
    nacheck = Rep.isnull().sum()
    #nacheck
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
    print('***read_data***',data)
    session, engine = connect_database()
    print('***session***', session)
    # Execute a SELECT query on the diabetes table
    query = text("SELECT * FROM "+data)
    result = session.execute(query)
    # Create a pandas DataFrame from the query result
    diabetes = pd.DataFrame(result.fetchall(), columns=result.keys())
    print('***diabetes***', diabetes.shape)
    print('***diabetes data:***', diabetes.head())
    print(diabetes)
    session.close()
    return diabetes


def train_model(data: str = 'Diabetes_clean'):
    print("***train_model***")
    clean_df = read_data(data)  # read data from data base
    # Let's store readmitted in y and rest of the columns in X,

    Y = clean_df['30readmit']
    X = clean_df.drop(['30readmit'], axis=1)
    X = pd.get_dummies(X)

    # Splitting the data into training and vallidation data sets. The training data will contain 80 % of the data and validation will contain remaining 20%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                        random_state=7, stratify=Y)

    print("shape of Xtrain,Xtest:", X_train.shape, X_test.shape)

    # Connect to MLflow
    mlflow.set_tracking_uri("http://10.43.102.111:5000")

    # Enable autologging in MLflow
    mlflow.autolog(log_model_signatures=True, log_input_examples=True)

    # Set up the data
    s = setup(clean_df, target='30readmit', transform_target=True, log_experiment=True,
              experiment_name='Experimento proyecto final MLOPS 2023')
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