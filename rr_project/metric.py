from numpy import ndarray
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from scipy.stats import ks_2samp


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



