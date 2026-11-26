import numpy as np


def calc_bias(estimates: np.ndarray, true_value: float) -> float | None:
    clean = estimates[~np.isnan(estimates)]
    if len(clean) == 0:
        return None
    return round(float(np.mean(clean) - true_value), 4)


def calc_std(estimates: np.ndarray) -> float | None:
    clean = estimates[~np.isnan(estimates)]
    if len(clean) == 0:
        return None
    return round(float(np.std(clean)), 4)


def calc_detection_rate(estimates: np.ndarray, threshold: float) -> float | None:
    clean = estimates[~np.isnan(estimates)]
    if len(clean) == 0:
        return None
    return round(float(np.mean(np.abs(clean) > threshold)), 4)


def calc_false_positive_rate(estimates: np.ndarray, threshold: float) -> float | None:
    clean = estimates[~np.isnan(estimates)]
    if len(clean) == 0:
        return None
    return round(float(np.mean(np.abs(clean) > threshold)), 4)
