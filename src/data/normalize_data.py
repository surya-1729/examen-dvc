import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_data():

    INPUT_DIR = "data/processed_data"
    OUTPUT_DIR = "data/processed_data"

    # Load train and test features
    X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"))

    # Initialize scaler
    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using same scaler
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame (keep column names)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Save scaled datasets
    X_train_scaled.to_csv(
        os.path.join(OUTPUT_DIR, "X_train_scaled.csv"), index=False
    )
    X_test_scaled.to_csv(
        os.path.join(OUTPUT_DIR, "X_test_scaled.csv"), index=False
    )

    print("✅ Normalized datasets saved in data/processed/")

if __name__ == "__main__":
    normalize_data()