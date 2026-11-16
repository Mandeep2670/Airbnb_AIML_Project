from fastapi.responses import FileResponse
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

from preprocessing import add_geographical_features

app = FastAPI(
    title="Airbnb Price Prediction API",
    description="Machine Learning API for Airbnb nightly price prediction",
    version="1.0",
)

# Load the trained model once
MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"

model = joblib.load(MODEL_PATH)


# Input validation schema
class AirbnbInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    neighbourhood_group: Literal[
        "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"
    ]

    room_type: Literal["Entire home/apt", "Private room", "Shared room"]

    latitude: float = Field(ge=40.0, le=41.0)
    longitude: float = Field(ge=-75.0, le=-73.0)

    minimum_nights: int = Field(ge=1)
    number_of_reviews: int = Field(ge=0)
    reviews_per_month: float = Field(ge=0)

    calculated_host_listings_count: int = Field(ge=1)
    availability_365: int = Field(ge=0, le=365)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_price(data: AirbnbInput):

    # Convert input into a DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Calculate distance to Times Square
    input_df = add_geographical_features(input_df)

    # Predict nightly price
    prediction = model.predict(input_df)[0]

    return {
        "predicted_price_usd": round(max(0, float(prediction)), 2),
        "currency": "USD",
        "model": type(model.named_steps["regressor"]).__name__,
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
