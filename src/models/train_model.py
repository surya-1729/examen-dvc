import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


def train_model():

    DATA_DIR = "data/processed_data"
    MODEL_DIR = "models"

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load training data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

    # Load best parameters
    params_path = os.path.join(MODEL_DIR, "best_params.pkl")
    with open(params_path, "rb") as f:
        best_params = pickle.load(f)

    # Initialize model with best parameters
    model = RandomForestRegressor(
        **best_params,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Save trained model
    model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved in models/trained_model.pkl")


if __name__ == "__main__":
    train_model()
