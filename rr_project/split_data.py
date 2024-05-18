from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rr_project.config.const import SEED


def split_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    random_state: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.

    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split feature and target matrices for training and testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return (
        X_train,
        X_test,
        y_train,
    )
