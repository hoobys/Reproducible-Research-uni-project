from sklearn.metrics import auc, roc_curve
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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
