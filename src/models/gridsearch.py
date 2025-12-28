import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def run_gridsearch():

    # Paths
    DATA_DIR = "data/processed_data"
    MODEL_DIR = "models"

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load scaled training data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

    # Define model
    model = RandomForestRegressor(random_state=42)

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    # GridSearch
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    # Fit
    grid_search.fit(X_train, y_train)

    # Best parameters
    best_params = grid_search.best_params_

    # Save best parameters
    output_path = os.path.join(MODEL_DIR, "best_params.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(best_params, f)

    print("✅ Best parameters saved to models/best_params.pkl")
    print("🏆 Best parameters:", best_params)


if __name__ == "__main__":
    run_gridsearch()
