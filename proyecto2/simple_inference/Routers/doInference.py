from Schemas.schemas import Covertype
import os
from fastapi import APIRouter, FastAPI, HTTPException
import numpy as np
import pandas as pd
from joblib import load
import numpy as np
import pandas as pd

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


def prepare_sample(covertype: Covertype): #prepare data before prediction
        _wilderness_area_map = {'Cache': 2, 'Commanche': 0, 'Rawah': 1,'Neota': 3,}
        _Soil_type_map = {'C2702':0,'C2703':1,'C2704':2,'C2705':3,'C2706':4,'C2717':5,'C3501':6,'C3502':7,'C4201':8,'C4703':9,'C4704':10,
                          'C4744':11,'C4758':12,'C5101':13,'C5151':14,'C6101':15,'C6102':16,'C6731':17,'C7101':18,'C7102':19,'C7103':20,
                          'C7201':21,'C7202':22,'C7700':23,'C7701':24,'C7702':25,'C7709':26,'C7710':27,'C7745':28,'C7746':29,
                          'C7755':30,'C7756':31,'C7757':32,'C7790':33,'C8703':34,'C8707':35,'C8708':36,'C8771':37,'C872':38, 'C876':39}

        wilderness = _wilderness_area_map[covertype.Wilderness_Area]
        soil = _Soil_type_map[covertype.Soil_Type]
        sample = [covertype.Elevation,covertype.Aspect,covertype.Slope,covertype.Horizontal_Distance_To_Hydrology,
                  covertype.Vertical_Distance_To_Hydrology,covertype.Horizontal_Distance_To_Roadways,covertype.Hillshade_9am,
                  covertype.Hillshade_Noon,covertype.Hillshade_3pm,covertype.Horizontal_Distance_To_Fire_Points,
                  wilderness, soil]
        sample = np.array([np.asarray(sample)]).reshape(-1, 1)

        return sample.reshape(1, -1)

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


def insert_data(covertype):
    try:
        print('***insert_data inference***')
        # Connect to the database
        session, engine = connect_database()
        print("***df covertype inference***", covertype.info())
        # Define the table schema
        Base = declarative_base()
        # Define covertype table model
        class Covertype(Base):
            __tablename__ = 'covertype_inference'
            id = Column(Integer, primary_key=True, index=True)
            Elevation= Column(Integer)
            Aspect = Column(Integer)
            Slope = Column(Integer)
            Horizontal_Distance_To_Hydrology = Column(Integer)
            Vertical_Distance_To_Hydrology = Column(Integer)
            Horizontal_Distance_To_Roadways = Column(Integer)
            Hillshade_9am = Column(Integer)
            Hillshade_Noon = Column(Integer)
            Hillshade_3pm=Column(Integer)
            Horizontal_Distance_To_Fire_Points = Column(Integer)
            Wilderness_Area=Column(String)
            Soil_Type = Column(String)
            Cover_Type = Column(Integer)

        print('Base',Base)
        # Create table if it doesn't exist
        Base.metadata.create_all(bind=engine)
        # Insert the data into the table
        print('Insert the data into the table')
        covertype_table = Table('covertype_inference',Base.metadata, autoload=True)
        # Print schema of the covertype table
        print("covertype_inference:",covertype_table)
        # Create a connection object
        print("*** example query results:",session.execute(text('SELECT * FROM covertype_inference order by id desc limit 10')))

        for i, row in covertype.iterrows():
            print("***i:",i)
            print("***row:", row)
            covertype = Covertype( Elevation= row['Elevation'],
                                    Aspect = row['Aspect'],
                                    Slope = row['Slope'],
                                    Horizontal_Distance_To_Hydrology = row['Horizontal_Distance_To_Hydrology'],
                                    Vertical_Distance_To_Hydrology = row['Vertical_Distance_To_Hydrology'],
                                    Horizontal_Distance_To_Roadways = row['Horizontal_Distance_To_Roadways'],
                                    Hillshade_9am = row['Hillshade_9am'],
                                    Hillshade_Noon = row['Hillshade_Noon'],
                                    Hillshade_3pm=row['Hillshade_3pm'],
                                    Horizontal_Distance_To_Fire_Points = row['Horizontal_Distance_To_Fire_Points'],
                                    Wilderness_Area= row['Wilderness_Area'],
                                    Soil_Type = row['Soil_Type'],
                                    Cover_Type = row['Cover_Type'])
            session.add(covertype)

        print("***session before commit***",session)
        session.commit()
        print("*** example query results:", session.execute(text('SELECT * FROM covertype_inference order by id desc limit 1')))
        session.close()
        result="insert OK"
    except Exception as e:
        print("ERROR:",e)
        result=e
    return result

@router.post("/covertype")
async def test_model(covertype: Covertype, model:str='covertype'):
    path = "/work2/"
    print("path:",path+model+'_model.joblib')
    if not os.path.isfile(path+model+'_model.joblib'):
        raise HTTPException(status_code=500, detail="Unkown model: "+ model+" Try to train model first.")
    print("loadign mdoel from:",path+model+'_model.joblib')
    print("covertype",covertype)
    print("covertype",type(covertype))

    preparedData = prepare_sample(covertype)
    df_original_data = pd.DataFrame(dict(covertype), index=[0])
    print("df_original_data", df_original_data)
    print("df_original_data", type(df_original_data))
    model_loaded = load(path+model+'_model.joblib')
    covertypes = model_loaded.predict(preparedData)
    df_original_data['Cover_Type'] = covertypes
    print("df before insert>>>>>",df_original_data)
    print("insert_data>>>>>",insert_data(df_original_data))
    return 'cover type is: ' + str(int(covertypes))



