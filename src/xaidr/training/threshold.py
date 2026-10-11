"""Detection threshold chosen on the validation split (D7)."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_curve


def youden_threshold(y_true: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Threshold maximising sensitivity + specificity - 1; predictions are ``score >= threshold``.

    Chosen once on validation data and then frozen for the test and external sets.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        raise ValueError("the Youden threshold needs both classes in the validation labels")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # the first ROC point is an artificial "predict nothing" threshold above every score
    best = int(np.argmax((tpr - fpr)[1:])) + 1
    return float(thresholds[best])
