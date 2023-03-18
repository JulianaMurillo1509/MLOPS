from fastapi import FastAPI

from Routers import doInference
import uvicorn 


app = FastAPI()
app.include_router(doInference.router)

if __name__ == "__main__":
    uvicorn.run("main:app", port=8001, reload=True)
