from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from rr_project.config import SEED


def feature_selection_xgb(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    num_features: int,
    random_state: int = SEED,
) -> Tuple[np.ndarray, List[int]]:
    """
    Perform feature selection using XGBoost and feature importance.

    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector.
    num_features (int): Number of top features to select based on importance.
    random_state (int): Random state for reproducibility.

    Returns:
    np.ndarray: Reduced feature matrix with selected features.
    list: List of selected feature names or indices.
    """
    xgb = XGBClassifier(
        random_state=random_state, use_label_encoder=False, eval_metric="logloss"
    )
    xgb.fit(X, y)

    feature_importances = xgb.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]

    selected_indices = sorted_indices[:num_features]
    selected_features = (
        X[:, selected_indices]
        if isinstance(X, np.ndarray)
        else X.iloc[:, selected_indices].values
    )

    return selected_features, selected_indices
