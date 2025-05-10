# evaluator.py
"""
evaluator.py
Functions to compute regression and ranking metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float]:
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    return mse, rmse

def top_k_accuracy(y_true: pd.Series, y_pred: pd.Series, k: int = 10) -> float:
    common = y_true.index.intersection(y_pred.index)
    true_vals = y_true.loc[common]
    pred_vals = y_pred.loc[common]
    top_true = set(true_vals.nlargest(k).index)
    top_pred = set(pred_vals.nlargest(k).index)
    return len(top_true & top_pred)/k
