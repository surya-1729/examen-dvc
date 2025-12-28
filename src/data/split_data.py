import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(test_size=0.2, random_state=42):

    INPUT_DATA_PATH = "data/raw_data/raw.csv"
    OUTPUT_DATA_PATH = "data/processed_data"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Load dataset
    df = pd.read_csv(INPUT_DATA_PATH)

    # Separate features and target
    X = df.drop(columns=["date", "silica_concentrate"])
    y = df["silica_concentrate"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save outputs
    X_train.to_csv(os.path.join(OUTPUT_DATA_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DATA_PATH, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DATA_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DATA_PATH, "y_test.csv"), index=False)

    print("✅ Data successfully split and saved in data/processed_data/")

if __name__ == "__main__":
    split_data()