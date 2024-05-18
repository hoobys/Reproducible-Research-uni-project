from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from rr_project.config.const import SEED


def save_model(pipeline: Pipeline, model_name: str) -> None:
    """
    Save the model to a .pkl file.

    Parameters:
    pipeline (Pipeline): scikit-learn pipeline containing the model and scaler.
    model_name (str): Name of the model for saving the .pkl file.

    Returns:
    None
    """
    joblib.dump(pipeline, f"{model_name}.pkl")
    print(f"Model {model_name} has been trained and saved successfully.")


def train_and_save_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
    random_state: int = SEED,
) -> None:
    """
    Train and save specified models using a scikit-learn pipeline.

    Parameters:
    X_train (np.ndarray): Training feature matrix.
    y_train (np.ndarray): Training target vector.
    scaler (StandardScaler): Scaler for feature standardization.

    Returns:
    None
    """
    models = {
        "logistic_regression_model": LogisticRegression(random_state=random_state),
        "decision_tree_model": DecisionTreeClassifier(random_state=random_state),
        "random_forest_model": RandomForestClassifier(random_state=random_state),
        "gradient_boosting_model": GradientBoostingClassifier(
            random_state=random_state
        ),
        "xgboost_model": XGBClassifier(
            random_state=random_state, use_label_encoder=False, eval_metric="logloss"
        ),
    }

    for model_name, model in models.items():
        pipeline = Pipeline([("scaler", scaler), ("model", model)])
        pipeline.fit(X_train, y_train)
        save_model(pipeline, model_name)
