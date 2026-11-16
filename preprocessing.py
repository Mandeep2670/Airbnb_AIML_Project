import pandas as pd


def clean_data(df):
    df = df.copy()

    df["name"] = df["name"].fillna("Unknown Listing")
    df["host_name"] = df["host_name"].fillna("Unknown Host")

    df = df[df["price"] > 0].copy()

    df = df.drop_duplicates(subset=["id"])

    df.loc[
        (df["number_of_reviews"] == 0) & (df["reviews_per_month"].isna()),
        "reviews_per_month",
    ] = 0

    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    return df


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

categorical_features = ["neighbourhood_group", "room_type"]

numerical_features = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "distance_to_times_square_km",
]


def create_preprocessor():

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


import numpy as np


def add_geographical_features(df):
    df = df.copy()

    # Times Square coordinates
    ref_lat = 40.7580
    ref_lon = -73.9855

    lat1 = np.radians(df["latitude"])
    lon1 = np.radians(df["longitude"])

    lat2 = np.radians(ref_lat)
    lon2 = np.radians(ref_lon)

    # Haversine formula
    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    df["distance_to_times_square_km"] = 6371 * c

    return df
