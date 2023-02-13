import requests
import os
from fastapi import HTTPException, APIRouter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import requests
import os
import pandas as pd
from joblib import dump


from sklearn.svm import SVC
from sklearn import metrics


router = APIRouter(
    prefix="/train_model",
    tags= ['train models']
)

url_wine='https://docs.google.com/uc?export=download&id=1ZsJWYHxcEdJQdb62diQf8o3fvXFawt1a'
url_house_price='https://docs.google.com/uc?export=download&id=1WsTJN-u4YRrPKqJTp8h8iUSAdfJC8_qn'

def get_data(data:str='wine'):
    if data=='wine' or data=='houseprice':
        if not os.path.isfile(data+'.csv'):
            url = url_wine if data == 'wine' else url_house_price
            r = requests.get(url, allow_redirects=True)
            open(data+'.csv', 'wb').write(r.content)
    else:
        raise HTTPException(status_code=500, detail="Unkown dataset: "+ data)

@router.get("/wine")
async def train_model(data:str='wine'):
    get_data()
    if not os.path.isfile(data+'.csv'):
        raise HTTPException(status_code=500, detail="Unkown dataset: "+ data)
    df = pd.read_csv('wine.csv')
    df.columns = df.columns.str.replace(' ', '_')
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = SVC()
    model.fit(X_train, y_train)
    expected_y  = y_test
    predicted_y = model.predict(X_test)
    model_metrics = metrics.classification_report(expected_y, predicted_y, output_dict=True,zero_division=1)
    dump(model, data+'_model.joblib')
    return model_metrics

@router.get("/house_price")
async def train_model(data:str='houseprice'):
    get_data()
    if not os.path.isfile(data+'.csv'):
        raise HTTPException(status_code=500, detail="Unkown dataset: "+ data)
    df = pd.read_csv('houseprice.csv')
    df.columns = df.columns.str.replace(' ', '_')
    X = df.drop(['date','price','street','city','statezip','country'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = LinearRegression()
    model.fit(X_train, y_train)
    expected_y  = y_test
    predicted_y = model.predict(X_test)
    model_metrics = metrics.r2_score(expected_y,predicted_y)
    dump(model, data+'_model.joblib')
    return model_metrics