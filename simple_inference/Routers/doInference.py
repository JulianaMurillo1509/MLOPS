from Schemas.schemas import Wine,House
import os
from fastapi import APIRouter, FastAPI, HTTPException

import pandas as pd
from joblib import load


app = FastAPI()
router = APIRouter(
    prefix="/do_inference",
    tags= ['do inference']
)

@router.post("/wine")
async def train_model(wine: Wine, model:str='wine'):
    if not os.path.isfile(model+'_model.joblib'):
        raise HTTPException(status_code=500, detail="Unkown model: "+ model+" Try to train model first.")
    model_loaded = load(model+'_model.joblib')
    return int(model_loaded.predict(pd.DataFrame([wine.dict()]))[0])

@router.post("/house_price")
async def train_model(house: House, model:str='house_price'):
    if not os.path.isfile(model+'_model.joblib'):
        raise HTTPException(status_code=500, detail="Unkown model: "+ model+" Try to train model first.")
    model_loaded = load(model+'_model.joblib')
    return int(model_loaded.predict(pd.DataFrame([house.dict()]))[0])