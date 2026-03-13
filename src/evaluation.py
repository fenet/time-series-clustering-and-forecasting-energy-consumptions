"""
Evaluation utilities: MAE, RMSE, MAPE.
Used to compare baseline vs. future experiments.
"""

import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    non_zero = y_true != 0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) /
                                y_true[non_zero])) * 100)
