from sklearn.metrics import auc, roc_curve
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


@dataclass
class ModelInfo:
    y_proba: np.ndarray
    y_true: np.ndarray
    model_name: str


def get_roc_for_multiple_models(*args: ModelInfo) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--")  # Random chance line
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")

    for model_info in args:
        fpr, tpr, _ = roc_curve(model_info.y_true, model_info.y_proba)
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_info.model_name} (area = {auc_score:.2f})")

    ax.legend(loc="lower right")
    return fig


def load_models_and_generate_roc(
    models_directory: str, X_test: np.ndarray, y_test: np.ndarray, suffix: str = ".pkl"
) -> plt.Figure:
    model_infos = []
    for file_name in os.listdir(models_directory):
        if file_name.endswith(suffix):
            model_path = os.path.join(models_directory, file_name)
            model = joblib.load(model_path)
            y_proba = model.predict_proba(X_test)[:, 1]
            model_name = file_name.replace(suffix, "")
            model_info = ModelInfo(
                y_proba=y_proba, y_true=y_test, model_name=model_name
            )
            model_infos.append(model_info)

    return get_roc_for_multiple_models(*model_infos)


def load_model_and_generate_confusion_matrix(
    model_path: str, X_test: np.ndarray, y_test: np.ndarray
) -> plt.Figure:
    """
    Load a model from the specified path and generate a confusion matrix plot.

    Parameters:
    model_path (str): Path to the model file.
    X_test (np.ndarray): Test feature matrix.
    y_test (np.ndarray): True labels for the test set.

    Returns:
    plt.Figure: Figure object containing the confusion matrix plot.
    """
    # Load the model
    model = joblib.load(model_path)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.show()

    return fig
