from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from .logging_utils import get_logger

logger = get_logger(__name__)

def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> Dict[str, float]:
    logger.info(f"Evaluating model: {model_name}")
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For safety, though all your models have predict_proba
        y_scores = model.decision_function(X_test)
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(
        f"{model_name} - Acc: {acc:.4f}, Prec: {prec:.4f}, "
        f"Rec: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}"
    )

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
    }

def get_confusion_and_report(
    model, X_test: pd.DataFrame, y_test: pd.Series, target_names=None
) -> Tuple[np.ndarray, str]:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    return cm, report

def compute_roc_pr_curves(model, X_test: pd.DataFrame, y_test: pd.Series):
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    return fpr, tpr, precision, recall