"""
This script builds the baseline pipeline for:
- Task 1: Clustering (2023 data)
- Task 2: Forecasting (cluster-level + dataset-level baseline)
It also evaluates forecasting accuracy using MAE and RMSE.
"""

from preprocessing import load_energy_data, extract_time_series, scale_time_series
from clustering import cluster_baseline, save_cluster_assignments
from forecasting import cluster_level_forecast, dataset_level_forecast
from evaluation import mae, rmse, mape


def main():
    print("=== Loading data ===")
    df23 = load_energy_data("data/energy_2023.csv")
    df24 = load_energy_data("data/energy_2024.csv")

    ids = df23["ID"].tolist()
    X23 = extract_time_series(df23).values

    print("=== Scaling for clustering ===")
    X23_scaled = scale_time_series(X23)

    print("=== Running baseline clustering (Task 1) ===")
    labels, sil = cluster_baseline(X23_scaled, n_clusters=4)
    print(f"Silhouette Score: {sil:.4f}")
    save_cluster_assignments(ids, labels, "clusters_output.csv")

    print("=== Cluster-level forecasting (Task 2a) ===")
    unique_clusters = sorted(set(labels))
    cluster_errors = {}

    for cl in unique_clusters:
        cl_ids = [cid for cid, lab in zip(ids, labels) if lab == cl]
        preds = cluster_level_forecast(cl_ids, df23, df24)

        # Compute errors per cluster
        mae_list = []
        rmse_list = []

        for cid, (pred_df, truth_df) in preds.items():
            y_pred = pred_df["yhat"].values
            y_true = truth_df["y"].values
            mae_list.append(mae(y_true, y_pred))
            rmse_list.append(rmse(y_true, y_pred))

        cluster_errors[cl] = {
            "MAE": sum(mae_list) / len(mae_list),
            "RMSE": sum(rmse_list) / len(rmse_list)
        }

    print("Cluster-level forecasting errors:")
    for cl, err in cluster_errors.items():
        print(f"Cluster {cl} → MAE={err['MAE']:.2f}, RMSE={err['RMSE']:.2f}")

    print("=== Dataset-level forecasting baseline (Task 2b) ===")
    pred_mean, truth_mean = dataset_level_forecast(ids, df23, df24)
    YP = pred_mean["yhat"].values
    YT = truth_mean["y"].values
    print(f"Dataset-level MAE: {mae(YT, YP):.2f}")
    print(f"Dataset-level RMSE: {rmse(YT, YP):.2f}")

    print("=== Baseline complete ===")


if __name__ == "__main__":
    main()
