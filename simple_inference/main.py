from fastapi import FastAPI

from Routers import doInference, trainModel
import uvicorn 


app = FastAPI()
app.include_router(trainModel.router)
app.include_router(doInference.router)

if __name__ == "__main__":
    uvicorn.run("main:app",port=8000 ,reload=True)
