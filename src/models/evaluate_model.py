import os
import json
import pickle
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model():

    DATA_DIR = "data/processed_data"
    MODEL_DIR = "models"
    METRICS_DIR = "metrics"
    OUTPUT_DATA_DIR = "data"

    os.makedirs(METRICS_DIR, exist_ok=True)

    # Load test data
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

    # Load trained model
    model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test)

    # Save predictions
    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })

    predictions_path = os.path.join(OUTPUT_DATA_DIR, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mean_squared_error": mse,
        "r2_score": r2
    }

    # Save metrics
    metrics_path = os.path.join(METRICS_DIR, "scores.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Evaluation completed")
    print("📊 Metrics saved to metrics/scores.json")
    print("📁 Predictions saved to data/predictions.csv")


if __name__ == "__main__":
    evaluate_model()
