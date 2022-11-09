import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from numpy.random import default_rng

np.random.seed(42)

rng = default_rng()


def crs(
    training_data,
    training_labels,
    num_clusters,
    img_per_cluster,
    metric_clustering="euclidean",
):

    if metric_clustering == "euclidean":
        cluster_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=num_clusters)),
            ]
        )
    elif metric_clustering == "cosine":
        cluster_pipe = Pipeline(
            [
                ("scaler", Normalizer()),
                ("kmeans", KMeans(n_clusters=num_clusters)),
            ]
        )

    divided_data = [
        training_data[training_labels == i]
        for i in range(min(training_labels), max(training_labels) + 1)
    ]

    group_labels_per_group = []
    for i in range(len(divided_data)):
        group_labels_per_group.append(
            cluster_pipe.fit_predict(
                divided_data[i].reshape(divided_data[i].shape[0], -1)
            )
        )

    grouped_training_data = [
        current_data[group_labels == i]
        for current_data, group_labels in zip(
            divided_data, group_labels_per_group
        )
        for i in range(num_clusters)
    ]

    chosen_images = []

    # Randomly select from clusters
    for current_cluster in grouped_training_data:
        if len(current_cluster) > img_per_cluster:
            chosen_images.extend(
                current_cluster[
                    rng.choice(
                        len(current_cluster),
                        size=img_per_cluster,
                        replace=False,
                    )
                ]
            )
        else:
            chosen_images.extend(current_cluster)

    return chosen_images
