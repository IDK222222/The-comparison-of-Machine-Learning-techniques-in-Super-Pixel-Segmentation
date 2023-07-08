import numpy as np
from sklearn.cluster import MiniBatchKMeans


def kmeans_super_pixellib(img, n_clusters):
    # Extract the relevant features from the image
    features = img.reshape(-1, 3).astype(np.float32)

    print("Started Kmeans")

    # Initialize the cluster centroids
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=0, n_init='auto').fit(features)

    # Assign each data point to the nearest centroid
    labels = kmeans.predict(features)

    # Assign the resulting clusters to the super pixels in the image
    super_pixels = labels.reshape(*img.shape[:2])
    print("Finished Kmeans")

    return super_pixels

def kmeans_super_pixel(img, n_clusters):
    print("Started Kmeans")
    # Extract the relevant features from the image
    features = img.reshape(-1, 3).astype(np.float32)

    # Initialize the cluster centroids
    np.random.seed(0)
    centroids = features[np.random.choice(features.shape[0], size=n_clusters, replace=False)]

    # Iteratively assign each data point to the nearest centroid and update the centroids
    max_iter = 10
    for i in range(max_iter):
        # Compute distances between each data point and each centroid
        distances = np.sqrt(np.sum(np.square(features[:, np.newaxis] - centroids), axis=2))

        # Assign each data point to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Update the centroids based on the points assigned to each cluster
        for j in range(n_clusters):
            cluster_features = features[labels == j]
            if len(cluster_features) > 0:
                centroids[j] = np.mean(cluster_features, axis=0)

    # Assign the resulting clusters to the super pixels in the image
    super_pixels = labels.reshape(*img.shape[:2])
    print("Finished Kmeans")
    return super_pixels