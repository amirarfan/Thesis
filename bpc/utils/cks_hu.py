#%%
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
from scipy.spatial.distance import cdist
import scipy.io as sio


def extract_samples_from_cluster(
    orig_data,
    clusters,
    cluster_centers,
    images_chosen_per_cluster,
    metric="euclidean",
):

    chosen_images = []

    for i in range(len(clusters)):

        if len(clusters[i]) <= images_chosen_per_cluster:
            chosen_images.append(clusters[i])
            continue
        curr_group = clusters[i].copy()

        cc = cluster_centers[i].reshape(1, -1)

        distances = cdist(cc, clusters[i], metric=metric)

        idx_max_dist = np.argmax(distances)

        idx_chosen = [idx_max_dist]

        idx_not_chosen = np.arange(clusters[i].shape[0])

        idx_not_chosen = np.delete(idx_not_chosen, idx_chosen, axis=0)

        curr_group = np.delete(curr_group, idx_chosen, axis=0)

        while len(idx_chosen) < images_chosen_per_cluster:

            distances = np.sum(
                cdist(clusters[i][idx_chosen], curr_group, metric=metric),
                axis=0,
            )

            idx_max_dist = np.argmax(distances)

            idx_chosen.append(idx_not_chosen[idx_max_dist])

            curr_group = np.delete(curr_group, idx_max_dist, axis=0)

            idx_not_chosen = np.delete(idx_not_chosen, idx_max_dist, axis=0)

        chosen_images.append(orig_data[i][idx_chosen])
    return chosen_images


def ks_variant(
    orig_data,
    training_data,
    training_labels,
    num_clusters,
    img_per_cluster,
    metric_clustering="euclidean",
    metric_ks="euclidean",
):

    if training_labels is None:

        raise ValueError(
            "Must Provide Training Labels for Within Group Clustering"
        )

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
    divided_orig_data = [
        orig_data[training_labels == i]
        for i in range(min(training_labels), max(training_labels) + 1)
    ]
    group_labels_per_group = []
    cluster_centers = []
    for i in range(len(divided_data)):
        group_labels_per_group.append(
            cluster_pipe.fit_predict(divided_data[i])
        )
        cluster_centers.extend(cluster_pipe["kmeans"].cluster_centers_)
        grouped_training_data = [
            current_data[group_labels == i]
            for current_data, group_labels in zip(
                divided_data, group_labels_per_group
            )
            for i in range(num_clusters)
        ]
        grouped_orig_data = [
            orig_data[group_labels == i]
            for orig_data, group_labels in zip(
                divided_orig_data, group_labels_per_group
            )
            for i in range(num_clusters)
        ]

    print("Done grouping")
    return extract_samples_from_cluster(
        grouped_orig_data,
        grouped_training_data,
        cluster_centers,
        img_per_cluster,
        metric=metric_ks,
    )


#%%
if __name__ == "__main__":
    data = sio.loadmat(
        r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\imageclassification_proj\lstsq_gpu\data\mnistdata.mat"
    )

    training_data = data["Xtrain"]

    training_labels = np.concatenate(data["Gtrain"])
    chosen_images = ks_variant(
        training_data,
        training_labels,
        10,
        5,
        within_group=True,
        metric_clustering="cosine",
        metric_ks="cosine",
    )

    chosen_images = np.concatenate(chosen_images)
    chosen_images = chosen_images.reshape(chosen_images.shape[0], 28, 28)
    np.save("ks_tot_20_mixdist.npy", chosen_images)
