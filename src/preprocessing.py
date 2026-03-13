"""
Preprocessing utilities for the energy forecasting project.
This file contains baseline-normalization and reshaping functions.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_energy_data(path: str) -> pd.DataFrame:
    """
    Load energy data from a CSV file where rows are customers
    and columns are dates (wide format).
    """
    df = pd.read_csv(path)
    df.columns = [c if c == "ID" else pd.to_datetime(c) for c in df.columns]
    return df


def extract_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the daily consumption values (drop ID column).
    """
    return df.drop(columns=["ID"])


def scale_time_series(X):
    """
    Row-wise standardization to make clustering focus on shape, not magnitude.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
