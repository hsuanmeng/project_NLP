from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
from model import predict, LDA, summary
import json

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/isintent/pred/{filename}")
async def getIntentresult(filename:str):
    result = predict.Run(filename)
    return json.loads(result)

@app.get("/outintent/categroy/{filename}")
async def getNotIntentresult(filename:str):
    result = LDA.Run(filename)
    return json.loads(result)

@app.get("/statistic/summary/{filename}")
async def getSummary(filename:str):
    result = summary.Run(filename)
    return json.loads(result)

if __name__ == "__main__":
    uvicorn.run("new_web:app", host="127.0.0.1", port=8000, log_level="info")