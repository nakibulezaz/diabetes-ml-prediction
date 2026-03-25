from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from .logging_utils import get_logger
from .config import RANDOM_SEED

logger = get_logger(__name__)

def split_features_target(df: pd.DataFrame, target_col: str = "Outcome") -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Splitting features and target")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Performing stratified train-test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}"
    )
    return X_train, X_test, y_train, y_test

def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    logger.info("Scaling features with StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    logger.info("Feature scaling complete")
    return X_train_scaled, X_test_scaled, scaler

def apply_smote(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_SEED,
    k_neighbors: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Applying SMOTE to balance classes")
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)
    X_balanced = pd.DataFrame(X_balanced, columns=X_train_scaled.columns)
    logger.info(
        f"After SMOTE - X_balanced: {X_balanced.shape}, y_balanced: {y_balanced.shape}"
    )
    return X_balanced, y_balanced