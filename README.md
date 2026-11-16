
# Airbnb Data Preprocessing and Exploratory Analysis

An advanced AI/ML project using the New York City Airbnb
2019 dataset.

## Features

- Data cleaning and missing-value handling
- Exploratory Data Analysis
- Feature engineering
- Airbnb nightly price prediction
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression
- MLflow experiment tracking
- FastAPI prediction service
- Automated testing with Pytest
- Docker containerization
- GitHub Actions CI/CD

## Dataset

Dataset: AB_NYC_2019.csv

Source: New York City Airbnb Open Data, 2019

## Technology Stack

Python, Pandas, NumPy, Scikit-learn,
Matplotlib, Seaborn, MLflow, FastAPI,
Pytest, Docker and GitHub Actions.

## Local Setup

Install dependencies:

    python -m pip install -r requirements.txt

Train models:

    python train.py

Run the API:

    uvicorn app:app --reload --port 8000

Run tests:

    python -m pytest -v

## API Endpoints

- GET / : Web Dashboard
- GET /health : Health Check
- POST /predict : Airbnb Price Prediction

## Model Evaluation

The models are compared using cross-validation
MAE, RMSE and R².

The final model is selected based on CV RMSE.
