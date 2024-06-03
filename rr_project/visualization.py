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
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.show()

    return fig


def plot_ks_curve(y_true, y_proba, ax, model_name):
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    diff = tpr - fpr
    ks_stat = np.max(diff)
    ks_threshold = thresholds[np.argmax(diff)]
    ks_index = np.argmax(diff)

    ax.plot(thresholds, tpr, label="Tpr")
    ax.plot(thresholds, fpr, label="Fpr")
    ax.plot(thresholds, diff, label="Diff")
    ax.scatter([ks_threshold], [tpr[ks_index]], color="red")
    ax.scatter([ks_threshold], [fpr[ks_index]], color="red")
    ax.plot(
        [ks_threshold, ks_threshold],
        [fpr[ks_index], tpr[ks_index]],
        "r-",
        lw=2,
        label=f"KS={ks_stat:.2f}",
    )
    ax.set_title(f"KS Curve - {model_name}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Value")
    ax.legend()


def load_models_and_generate_ks_curve(
    models_directory: str,
    X_test_iv: np.ndarray,
    y_test_iv: np.ndarray,
    X_test_xgb: np.ndarray,
    y_test_xgb: np.ndarray,
    iv_model_name: str = "XGBClassifier_model_iv_cv.pkl",
    xgb_model_name: str = "GradientBoostingClassifier_model_xgb_cv.pkl",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, (feature_type, X_test, y_test, model_name) in zip(
        axes,
        [
            ("IV Feature Selection", X_test_iv, y_test_iv, iv_model_name),
            ("XGB Feature Selection", X_test_xgb, y_test_xgb, xgb_model_name),
        ],
    ):
        model_path = os.path.join(models_directory, model_name)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_proba = model.predict_proba(X_test)[:, 1]
            plot_ks_curve(y_test, y_proba, ax, feature_type)

    plt.tight_layout()
    plt.show()
    return fig
