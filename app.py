# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("salary_predictor.pkl")

class SalaryInput(BaseModel):
    years_experience: float
    age: float

@app.post("/predict")
def predict_salary(input_data: SalaryInput):
    if input_data.years_experience is None or input_data.age is None:
        raise HTTPException(status_code=400, detail="Missing required fields")

    features = np.array([[input_data.years_experience, input_data.age]])
    prediction = model.predict(features)[0]

    return {
        "input": {
            "years_experience": input_data.years_experience,
            "age": input_data.age
        },
        "predicted_salary": round(float(prediction), 2)
    }
