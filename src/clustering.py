"""
Clustering utilities (Task 1).
We use TimeSeriesKMeans (Euclidean) as the baseline.
"""

import pandas as pd
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans


def cluster_baseline(X_scaled, n_clusters: int = 4, seed: int = 42):
    """
    Baseline clustering using TimeSeriesKMeans with Euclidean distance.
    Returns: labels array, silhouette score.
    """
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="euclidean",
        random_state=seed
    )
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return labels, sil


def save_cluster_assignments(ids, labels, path):
    """
    Save ID -> cluster mapping to CSV.
    """
    df = pd.DataFrame({"ID": ids, "cluster": labels})
    df.to_csv(path, index=False)
