from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib 
import numpy as np

app = FastAPI()

model = joblib.load('car_price_model_onehot.pkl')
scaler = joblib.load('scaler_onehot.pkl')
columns = joblib.load('onehot_columns.pkl')

class carInput(BaseModel):
    model_name : str
    year : int
    transmission : str
    mileage : int
    fuelType : str
    tax : int
    mpg : float
    engineSize : float


@app.get('/')

def home():
    return{'message': 'Car Price Prediction API is running!'}

@app.post('/predict')

def predict(data: carInput):
    # Step 1: Dict banao
    input_dict = {
        "year": data.year,
        "mileage": data.mileage,
        "tax": data.tax,
        "mpg": data.mpg,
        "engineSize": data.engineSize,
        f"model_{data.model_name}": 1,
        f"transmission_{data.transmission}": 1,
        f"fuelType_{data.fuelType}": 1,
    }

    input_df= pd.DataFrame([input_dict])
    input_df= input_df.reindex(columns=columns, fill_value=0)

    numeric_cols = ['year', 'mileage', 'tax', 'mpg']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    price = model.predict(input_df)[0]
    return{'predicted_price': round(float(price),2)}