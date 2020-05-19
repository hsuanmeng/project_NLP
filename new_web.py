from fastapi import FastAPI
import uvicorn
from model import predict
from model import LDA

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/isintent/pred/{filename}")
def getIntentresult(filename:str):
    result = predict.Run(filename)
    return result

@app.get("/outintent/categroy/{filename}")
def getNotIntentresult(filename:str):
    result = LDA.Run(filename)
    return result

if __name__ == "__main__":
    uvicorn.run("new_web:app", host="127.0.0.1", port=8000, log_level="info")