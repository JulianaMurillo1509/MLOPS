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
    if not os.path.isfile(model+'_model.joblib'):
        raise HTTPException(status_code=500, detail="Unkown model: "+ model+" Try to train model first.")
    model_loaded = load(model+'_model.joblib')
    return int(model_loaded.predict(pd.DataFrame([penguin.dict()]))[0])



