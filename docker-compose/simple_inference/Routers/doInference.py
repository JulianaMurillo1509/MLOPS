from Schemas.schemas import Penguin
import os
from fastapi import APIRouter, FastAPI, HTTPException
import numpy as np
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


def prepare_sample(penguin: Penguin): #prepare data before prediction
        _island_map = {'Torgersen': 2, 'Biscoe': 0, 'Dream': 1}
        _sex_map = {'male': 2, 'female': 1}

        island = _island_map[penguin.island]
        sex = _sex_map[penguin.sex]
        sample = [penguin.bill_length_mm, penguin.bill_depth_mm, penguin.flipper_length_mm, penguin.body_mass_g, island, sex,penguin.year]
        sample = np.array([np.asarray(sample)]).reshape(-1, 1)

        return sample.reshape(1, -1)

@router.post("/penguins")
async def test_model(penguin: Penguin, model:str='penguins'):
    path = "/work2/"
    print("path:",path+model+'_model.joblib')
    if not os.path.isfile(path+model+'_model.joblib'):
        raise HTTPException(status_code=500, detail="Unkown model: "+ model+" Try to train model first.")
    print("loadign mdoel from:",path+model+'_model.joblib')
 
    spicies_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    preparedData = prepare_sample(penguin)
    model_loaded = load(path+model+'_model.joblib')
    species = model_loaded.predict(preparedData)[0]
    result = (spicies_map[species])
    answer = "the penguin specie is:" + result
    return answer



