from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from rr_project.config.const import SEED


def feature_selection_xgb(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    importance_threshold: float = 0.01,
    random_state: int = SEED,
) -> List[str]:
    """
    Perform feature selection using XGBoost and feature importance.

    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector.
    importance_threshold (float): Threshold for feature importance to select features.
    random_state (int): Random state for reproducibility.

    Returns:
    list: List of selected feature names.
    """
    xgb = XGBClassifier(
        random_state=random_state, use_label_encoder=False, eval_metric="logloss"
    )
    xgb.fit(X, y)

    feature_importances = xgb.feature_importances_
    selected_indices = np.where(feature_importances >= importance_threshold)[0]

    if isinstance(X, np.ndarray):
        feature_names = selected_indices.tolist()
    else:
        feature_names = X.columns[selected_indices].tolist()

    return feature_names
