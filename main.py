from src.logging_utils import get_logger
from src.config import RAW_DATA_PATH, FIGURES_DIR
from src.data_loader import load_raw_data
from src.preprocessing import (
    split_features_target,
    stratified_train_test_split,
    scale_features,
    apply_smote,
)
from src.modeling import (
    get_base_models,
    tune_gradient_boosting,
    tune_random_forest,
    tune_xgboost,
    cross_validate_models,
    build_voting_classifier,
    build_stacking_classifier,
)
from src.evaluation import evaluate_model
from src.visualization import (
    plot_class_distribution,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_roc_curves,
)

import pandas as pd

logger = get_logger("main")

def run_pipeline():
    logger.info("Starting end-to-end diabetes prediction pipeline")

    # Load data
    df = load_raw_data(RAW_DATA_PATH)

    # EDA figures
    plot_class_distribution(df["Outcome"], FIGURES_DIR / "class_distribution.png")
    plot_feature_distributions(df.drop(columns=["Outcome"]), FIGURES_DIR / "feature_distributions.png")
    plot_correlation_matrix(df, FIGURES_DIR / "correlation_matrix.png")

    # Preprocessing
    X, y = split_features_target(df, target_col="Outcome")
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)

    # Models
    base_models = get_base_models()
    gb_best = tune_gradient_boosting(X_train_balanced, y_train_balanced)
    rf_best = tune_random_forest(X_train_balanced, y_train_balanced)
    xgb_best = tune_xgboost(X_train_balanced, y_train_balanced)

    base_models["Gradient Boosting"] = gb_best
    base_models["Random Forest"] = rf_best
    base_models["XGBoost"] = xgb_best

    # Cross-validation
    cv_results = cross_validate_models(base_models, X_train_balanced, y_train_balanced)
    logger.info(f"\nCross-validation results:\n{cv_results}")

    # Train and evaluate on test set
    results = []
    roc_curves = {}

    for name, model in base_models.items():
        logger.info(f"Training model: {name}")
        model.fit(X_train_balanced, y_train_balanced)
        metrics = evaluate_model(model, X_test_scaled, y_test, model_name=name)
        results.append(metrics)

        from src.evaluation import compute_roc_pr_curves
        fpr, tpr, precision, recall = compute_roc_pr_curves(model, X_test_scaled, y_test)
        roc_curves[name] = {"fpr": fpr, "tpr": tpr}

    results_df = pd.DataFrame(results)
    logger.info(f"\nTest set results:\n{results_df}")

    # ROC curves
    plot_roc_curves(roc_curves, FIGURES_DIR / "roc_curves.png")

    # Ensembles
    lr_model = base_models["Logistic Regression"]
    svm_model = base_models["SVM (RBF)"]

    voting_clf = build_voting_classifier(gb_best, rf_best, lr_model, svm_model, xgb_best)
    voting_clf.fit(X_train_balanced, y_train_balanced)
    voting_metrics = evaluate_model(voting_clf, X_test_scaled, y_test, "Voting Classifier")

    stacking_clf = build_stacking_classifier(gb_best, rf_best, svm_model, xgb_best)
    stacking_clf.fit(X_train_balanced, y_train_balanced)
    stacking_metrics = evaluate_model(stacking_clf, X_test_scaled, y_test, "Stacking Classifier")

    logger.info(f"Voting classifier metrics: {voting_metrics}")
    logger.info(f"Stacking classifier metrics: {stacking_metrics}")

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()