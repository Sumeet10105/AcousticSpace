import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.dataset import load_split


def compute_eer(y_true, y_scores):
    """
    Computes Equal Error Rate (EER) from true labels and predicted scores.
    y_true: 1 = real, 0 = fake
    y_scores: model's predicted probability of being 'real'
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Find the threshold where FPR and FNR are closest (crossing point)
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, thresholds[idx]


if __name__ == "__main__":
    X_train, y_train = load_split("train")
    X_dev, y_dev = load_split("dev")

    # Class weighting: 'balanced' automatically weights classes
    # inversely proportional to their frequency
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    # Predicted probability of being 'real' (class 1)
    dev_scores = model.predict_proba(X_dev)[:, 1]

    eer, threshold = compute_eer(y_dev, dev_scores)
    print(f"Dev EER: {eer * 100:.2f}%  (threshold: {threshold:.4f})")