import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import clean_data, add_geographical_features
from model import create_model


def train_models(args):

    # 1. Configure MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Airbnb_Price_Prediction")

    # 2. Load and clean the dataset
    print("Loading Airbnb Dataset...")

    df = pd.read_csv("AB_NYC_2019.csv")
    df = clean_data(df)
    df = add_geographical_features(df)

    print("Cleaned Dataset Shape:", df.shape)

    # 3. Select input features and target
    features = [
        "neighbourhood_group",
        "room_type",
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "distance_to_times_square_km",
    ]

    X = df[features].copy()
    y = df["price"].copy()

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    # 5. Define algorithms
    model_names = ["linear_regression", "random_forest", "gradient_boosting"]

    results = []
    run_ids = {}

    # 6. Train and track each model
    for name in model_names:

        print(f"\nTraining {name}...")

        model = create_model(name)

        if name == "random_forest":
            model.set_params(
                regressor__n_estimators=args.n_estimators,
                regressor__max_depth=args.max_depth,
            )

        elif name == "gradient_boosting":
            model.set_params(
                regressor__n_estimators=args.n_estimators,
                regressor__max_depth=args.max_depth,
                regressor__learning_rate=args.learning_rate,
            )

        with mlflow.start_run(run_name=name) as run:

            run_ids[name] = run.info.run_id

            # Log hyperparameters
            mlflow.log_param("algorithm", name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("cv_folds", 3)
            mlflow.log_param("random_state", 42)

            if name != "linear_regression":
                mlflow.log_param("n_estimators", args.n_estimators)
                mlflow.log_param("max_depth", args.max_depth)

            if name == "gradient_boosting":
                mlflow.log_param("learning_rate", args.learning_rate)

            # Cross-validation on training data only
            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                cv=3,
                scoring={
                    "mae": "neg_mean_absolute_error",
                    "rmse": "neg_root_mean_squared_error",
                    "r2": "r2",
                },
                n_jobs=1,
            )

            mae = -cv_results["test_mae"].mean()
            rmse = -cv_results["test_rmse"].mean()
            r2 = cv_results["test_r2"].mean()

            # Save metrics in MLflow
            mlflow.log_metrics(
                {"cv_mae": float(mae), "cv_rmse": float(rmse), "cv_r2": float(r2)}
            )

            results.append(
                {
                    "Model": name,
                    "CV_MAE": round(mae, 2),
                    "CV_RMSE": round(rmse, 2),
                    "CV_R2": round(r2, 4),
                }
            )

            print(f"{name} completed!")
            print("CV MAE:", round(mae, 2))
            print("CV RMSE:", round(rmse, 2))
            print("CV R2:", round(r2, 4))

    # 7. Compare models
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by="CV_RMSE", ascending=True)

    print("\n===================================")
    print("MODEL COMPARISON RESULTS")
    print("===================================")
    print(results_df.to_string(index=False))

    # 8. Select the best model
    best_model_name = results_df.iloc[0]["Model"]

    print("\nBest Model:", best_model_name)

    best_model = create_model(best_model_name)

    if best_model_name == "random_forest":
        best_model.set_params(
            regressor__n_estimators=args.n_estimators,
            regressor__max_depth=args.max_depth,
        )

    elif best_model_name == "gradient_boosting":
        best_model.set_params(
            regressor__n_estimators=args.n_estimators,
            regressor__max_depth=args.max_depth,
            regressor__learning_rate=args.learning_rate,
        )

    # 9. Train selected model on the training set
    best_model.fit(X_train, y_train)

    # 10. Evaluate once on held-out test data
    predictions = best_model.predict(X_test)

    test_mae = mean_absolute_error(y_test, predictions)

    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    test_r2 = r2_score(y_test, predictions)

    print("\n===================================")
    print("FINAL TEST SET PERFORMANCE")
    print("===================================")

    print("MAE:", round(test_mae, 2))
    print("RMSE:", round(test_rmse, 2))
    print("R2 Score:", round(test_r2, 4))

    # 11. Save the best trained model
    Path("model").mkdir(exist_ok=True)

    model_path = "model/model.joblib"

    joblib.dump(best_model, model_path)

    # 12. Save comparison results
    results_df.to_csv("model_comparison.csv", index=False)

    # 13. Log the selected model and final metrics
    with mlflow.start_run(run_id=run_ids[best_model_name]):

        mlflow.log_metrics(
            {
                "test_mae": float(test_mae),
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2),
            }
        )

        mlflow.log_artifact(model_path)
        mlflow.log_artifact("model_comparison.csv")

    print("\nBest model saved:", model_path)
    print("Model comparison saved: model_comparison.csv")
    print("MLflow experiments recorded successfully!")
    print("Training completed!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Airbnb price prediction models")

    parser.add_argument("--n_estimators", type=int, default=100)

    parser.add_argument("--max_depth", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=0.1)

    args = parser.parse_args()

    if args.n_estimators < 1:
        parser.error("--n_estimators must be positive")

    if args.max_depth < 1:
        parser.error("--max_depth must be positive")

    if not 0 < args.learning_rate <= 1:
        parser.error("--learning_rate must be between 0 and 1")

    train_models(args)
