from typing import Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from .logging_utils import get_logger
from .config import RANDOM_SEED

logger = get_logger(__name__)

def get_base_models() -> Dict[str, object]:
    logger.info("Initializing base models")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }
    return models

def tune_gradient_boosting(X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
    logger.info("Tuning Gradient Boosting hyperparameters")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.8, 0.9, 1.0],
    }
    gb = GradientBoostingClassifier(random_state=RANDOM_SEED)
    grid = GridSearchCV(
        gb,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)
    logger.info(f"Best GB params: {grid.best_params_}, ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_

def tune_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    logger.info("Tuning Random Forest hyperparameters")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)
    grid = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)
    logger.info(f"Best RF params: {grid.best_params_}, ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_

def tune_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    logger.info("Tuning XGBoost hyperparameters")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }
    xgb = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)
    logger.info(f"Best XGB params: {grid.best_params_}, ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_

def cross_validate_models(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    logger.info("Running cross-validation for all models")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for name, model in models.items():
        cv_acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        cv_auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
        rows.append(
            {
                "Model": name,
                "CV Accuracy Mean": cv_acc.mean(),
                "CV Accuracy Std": cv_acc.std(),
                "CV ROC-AUC Mean": cv_auc.mean(),
                "CV ROC-AUC Std": cv_auc.std(),
            }
        )
        logger.info(
            f"{name}: Acc {cv_acc.mean():.4f}±{cv_acc.std():.4f}, "
            f"ROC-AUC {cv_auc.mean():.4f}±{cv_auc.std():.4f}"
        )
    return pd.DataFrame(rows)

def build_voting_classifier(
    gb_best, rf_best, lr_model, svm_model, xgb_best
) -> VotingClassifier:
    logger.info("Building soft voting classifier")
    estimators = [
        ("gb", gb_best),
        ("rf", rf_best),
        ("lr", lr_model),
        ("svm", svm_model),
        ("xgb", xgb_best),
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting="soft")
    return voting_clf

def build_stacking_classifier(
    gb_best, rf_best, svm_model, xgb_best
) -> StackingClassifier:
    logger.info("Building stacking classifier")
    estimators = [
        ("gb", gb_best),
        ("rf", rf_best),
        ("svm", svm_model),
        ("xgb", xgb_best),
    ]
    final_estimator = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
    )
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=False,
    )
    return stacking_clf