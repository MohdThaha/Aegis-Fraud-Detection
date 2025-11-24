from typing import Dict
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(y) -> Dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}
