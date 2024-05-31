from numpy import ndarray
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from scipy.stats import ks_2samp
from typing import Tuple
import os
import joblib
import numpy as np
import pandas as pd


def calculate_accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    """Calculate accuracy of predictions."""
    return accuracy_score(y_true, y_pred)


def calculate_kappa(y_true: ndarray, y_pred: ndarray) -> float:
    """Calculate Cohen's Kappa statistic for the given predictions."""
    return cohen_kappa_score(y_true, y_pred)


def calculate_auc(y_true: ndarray, y_proba: ndarray) -> float:
    """Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    return roc_auc_score(y_true, y_proba)


def calculate_ks(y_true: ndarray, y_proba: ndarray) -> float:
    """Calculate the Kolmogorov-Smirnov statistic for the model's probability predictions."""
    pos_proba = y_proba[y_true == 1]
    neg_proba = y_proba[y_true == 0]
    return ks_2samp(pos_proba, neg_proba).statistic


def calculate_metrics(
    y_true: ndarray, y_pred: ndarray, y_proba: ndarray
) -> dict[str, float]:
    """Calculate and return a dictionary of various performance metrics."""
    accuracy = calculate_accuracy(y_true, y_pred)
    kappa = calculate_kappa(y_true, y_pred)
    auc = calculate_auc(y_true, y_proba)
    ks = calculate_ks(y_true, y_proba)

    return {"Accuracy": accuracy, "Kappa": kappa, "AUC": auc, "KS": ks}


def generate_performance_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    return metrics


def load_models_and_generate_tables(
    models_directory: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_xgb: np.ndarray,
    y_test_xgb: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_names = [
        "xgboost_model",
        "logistic_regression_model",
        "decision_tree_model",
        "random_forest_model",
        "gradient_boosting_model",
    ]

    model_names_xgb = [model_name + "_xgb" for model_name in model_names]

    data_iv_feature_selection = {
        "Model": model_names,
        "Accuracy": [],
        "Kappa": [],
        "AUC": [],
        "KS": [],
    }
    data_xgb_feature_selection = {
        "Model": model_names_xgb,
        "Accuracy": [],
        "Kappa": [],
        "AUC": [],
        "KS": [],
    }

    for model_name, model_name_xgb in zip(model_names, model_names_xgb):
        model_path_iv = os.path.join(models_directory, model_name + ".pkl")
        model_path_xgb = os.path.join(models_directory, model_name_xgb + ".pkl")

        if os.path.exists(model_path_iv):
            model_iv = joblib.load(model_path_iv)
            metrics = generate_performance_metrics(model_iv, X_test, y_test)
            for key in data_iv_feature_selection.keys():
                if key != "Model":
                    data_iv_feature_selection[key].append(metrics[key])

        if os.path.exists(model_path_xgb):
            model_xgb = joblib.load(model_path_xgb)
            metrics = generate_performance_metrics(model_xgb, X_test_xgb, y_test_xgb)
            for key in data_xgb_feature_selection.keys():
                if key != "Model":
                    data_xgb_feature_selection[key].append(metrics[key])

    table_iv = pd.DataFrame(data_iv_feature_selection)
    table_xgb = pd.DataFrame(data_xgb_feature_selection)

    return table_iv, table_xgb
