from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from preprocessing import create_preprocessor


def create_model(model_name="random_forest"):

    if model_name == "linear_regression":
        model = LinearRegression()

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline(
        steps=[("preprocessor", create_preprocessor()), ("regressor", model)]
    )

    return pipeline
