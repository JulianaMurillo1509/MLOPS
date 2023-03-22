from Schemas.schemas import Penguin
import os
from fastapi import APIRouter, FastAPI, HTTPException

import pandas as pd
from joblib import load


app = FastAPI()
router = APIRouter(
    prefix="/do_inference",
    tags= ['do inference']
)

@app.get("/")
async def root():
    return {"message": "Hello World inference"}

@router.post("/penguins")
async def test_model(penguin: Penguin, model:str='penguins'):
    path = "/work2/"
    print("path:",path+model+'_model.joblib')
    if not os.path.isfile(path+model+'_model.joblib'):
        raise HTTPException(status_code=500, detail="Unkown model: "+ model+" Try to train model first.")
    print("loadign mdoel from:",path+model+'_model.joblib')
    model_loaded = load(path+model+'_model.joblib')
    return int(model_loaded.predict(pd.DataFrame([penguin.dict()]))[0])



