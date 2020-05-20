from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
from model import predict, LDA, summary

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/isintent/pred/{filename}")
async def getIntentresult(filename:str):
    result = predict.Run(filename)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.get("/outintent/categroy/{filename}")
async def getNotIntentresult(filename:str):
    result = LDA.Run(filename)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.get("/statistic/summary/{filname}")
async def getSummary(filename:str):
    result = summary.Run(filename)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

if __name__ == "__main__":
    uvicorn.run("new_web:app", host="127.0.0.1", port=8000, log_level="info")