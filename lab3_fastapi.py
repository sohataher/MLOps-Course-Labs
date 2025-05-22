from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

logger.info("FastAPI app has started and is logging")


# Load model and transformer
model = joblib.load("xgboost_model.pkl")
transformer = joblib.load("column_transformer.pkl")

app = FastAPI()

class InputData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    return {"message": "Welcome to Churn Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    logging.info(f"Received input: {data}")
    input_df = [data.dict()]
    X_transformed = transformer.transform(pd.DataFrame(input_df))
    prediction = model.predict(X_transformed)[0]
    logging.info(f"Prediction: {prediction}")
    return {"prediction": int(prediction)}
