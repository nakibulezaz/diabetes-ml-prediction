from pathlib import Path
from src.config import RAW_DATA_PATH, PROCESSED_DIR
from src.data_loader import load_raw_data
from src.preprocessing import (
    split_features_target,
    stratified_train_test_split,
    scale_features,
    apply_smote,
)
from src.logging_utils import get_logger

logger = get_logger("generate_processed")

def main():
    logger.info("Starting processed data generation pipeline")

    df = load_raw_data(RAW_DATA_PATH)
    X, y = split_features_target(df, target_col="Outcome")
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_csv(PROCESSED_DIR / "X_train_scaled.csv", index=True)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test_scaled.csv", index=True)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=True)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=True)
    X_train_balanced.to_csv(PROCESSED_DIR / "X_train_balanced.csv", index=False)
    y_train_balanced.to_csv(PROCESSED_DIR / "y_train_balanced.csv", index=False)

    logger.info("Processed data saved successfully")

if __name__ == "__main__":
    main()