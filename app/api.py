from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()
model = pickle.load(open("../models/xgboost_model.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    input_data = np.array([data["features"]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]
    return {"risk_score": prediction}
