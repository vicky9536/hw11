from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

class Features(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(input: Features):
    pred = model.predict([input.features])
    return {"prediction": int(pred[0])}