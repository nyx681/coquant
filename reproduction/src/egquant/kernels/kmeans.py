import numpy as np
from sklearn.cluster import KMeans

def kmeans_fit(row_data):
    weights_np, sample_weight, n_cluster = row_data

    def fit(current_sample_weight):
        return KMeans(n_clusters=n_cluster, random_state=0, n_init='auto', max_iter=50).fit(weights_np, sample_weight=current_sample_weight)
    kmeans = fit(sample_weight)
    if not np.all(np.isfinite(kmeans.cluster_centers_)):
        active = weights_np.reshape(-1) != 0
        fallback_weight = active.astype(np.float32)
        if np.count_nonzero(fallback_weight) < n_cluster:
            fallback_weight = np.ones(weights_np.shape[0], dtype=np.float32)
        kmeans = fit(fallback_weight)
        if not np.all(np.isfinite(kmeans.cluster_centers_)):
            raise ValueError('K-means produced non-finite centers after fallback')
    return (kmeans.cluster_centers_.reshape(-1), kmeans.labels_.astype(np.int8, copy=False))
