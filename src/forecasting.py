"""
Forecasting utilities for Task 2.
Cluster-level and dataset-level baseline forecasting using Prophet.
"""

import pandas as pd
from prophet import Prophet


def prepare_prophet_format(row: pd.Series) -> pd.DataFrame:
    """
    Convert a single customer's row (wide format) into Prophet's long format.
    """
    df = pd.DataFrame({
        "ds": row.index,
        "y": row.values
    })
    return df


def forecast_customer_2024(row_2023, row_2024):
    """
    Train Prophet on 2023 and forecast 2024.
    Returns prediction DataFrame and MAE.
    """
    df_train = prepare_prophet_format(row_2023)
    df_test = prepare_prophet_format(row_2024)

    model = Prophet()
    model.fit(df_train)

    future = model.make_future_dataframe(periods=len(df_test), freq="D")
    forecast = model.predict(future)

    pred = forecast[forecast["ds"].isin(df_test["ds"])]
    return pred, df_test


def cluster_level_forecast(cluster_ids, df_2023, df_2024):
    """
    Forecast each customer in a cluster with a single Prophet model.
    """
    all_preds = {}
    for cid in cluster_ids:
        row23 = df_2023[df_2023["ID"] == cid].drop(columns=["ID"]).T.iloc[:, 0]
        row24 = df_2024[df_2024["ID"] == cid].drop(columns=["ID"]).T.iloc[:, 0]
        pred, truth = forecast_customer_2024(row23, row24)
        all_preds[cid] = (pred, truth)
    return all_preds


def dataset_level_forecast(all_ids, df_2023, df_2024):
    """
    Baseline model: one global model for the entire dataset.
    Train on aggregated mean profile.
    """
    row23_mean = df_2023.drop(columns=["ID"]).mean(axis=0)
    row24_mean = df_2024.drop(columns=["ID"]).mean(axis=0)

    pred_mean, truth_mean = forecast_customer_2024(row23_mean, row24_mean)
    return pred_mean, truth_mean
