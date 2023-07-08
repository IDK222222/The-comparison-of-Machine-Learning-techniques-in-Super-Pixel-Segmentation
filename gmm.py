import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def gmm_super_pixellib(img, n_components):
    print("Started gmm")
    # Extract the relevant features from the image
    features = img.reshape(-1, 3)

    # Estimate the parameters of the Gaussian distributions
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          n_init=1, max_iter=100, tol=1e-6).fit(features)

    # Assign each data point to the distribution with the highest likelihood
    labels = gmm.predict(features)

    # Assign the resulting clusters to the super pixels in the image
    super_pixels = labels.reshape(*img.shape[:2])
    print("finished gmm")
    return super_pixels


def gmm_super_pixel(img, n_components):
    print("Started gmm")
    # Extract the relevant features from the image
    features = img.reshape(-1, 3)

    # Initialize the means, covariances, and weights
    kmeans = KMeans(n_clusters=n_components, max_iter=10, tol=1e-8, n_init=1)
    labels = kmeans.fit_predict(features)
    means = kmeans.cluster_centers_
    covariances = np.zeros((n_components, 3, 3))
    weights = np.zeros(n_components)
    for i in range(n_components):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            weights[i] = len(indices) / len(features)
            covariances[i] = np.cov(features[indices].T)
        else:
            weights[i] = 0
            covariances[i] = np.eye(3)

    # Initialize the responsibilities
    responsibilities = np.zeros((len(features), n_components))

    # Run the EM algorithm to estimate the parameters of the Gaussian mixture model
    prev_likelihood = -np.inf
    for iteration in range(100):
        # E-step: compute the responsibilities
        for j in range(n_components):
            responsibilities[:, j] = weights[j] * multivariate_normal.pdf(features, means[j], covariances[j],
                                                                          allow_singular=True)
        responsibilities /= np.sum(responsibilities, axis=1)[:, np.newaxis]

        # M-step: update the parameters
        for j in range(n_components):
            sum_resp = np.sum(responsibilities[:, j])
            if sum_resp > 0:
                weights[j] = sum_resp / len(features)
                means[j] = np.sum(responsibilities[:, j, np.newaxis] * features, axis=0) / sum_resp
                centered = features - means[j]
                covariances[j] = np.dot(responsibilities[:, j] * centered.T, centered) / sum_resp + 1e-6 * np.eye(3)
            else:
                weights[j] = 0
                means[j] = np.random.rand(3)
                covariances[j] = np.eye(3)

        # Compute the likelihood and check for convergence
        log_likelihood = np.sum(np.log(np.sum(
            weights[j] * multivariate_normal.pdf(features, means[j], covariances[j], allow_singular=True) for j in
            range(n_components))))
        if np.abs(log_likelihood - prev_likelihood) < 1e-6:
            break
        prev_likelihood = log_likelihood

    # Assign each data point to the distribution with the highest likelihood
    labels = np.argmax(responsibilities, axis=1)

    # Assign the resulting clusters to the super pixels in the image
    super_pixels = labels.reshape(*img.shape[:2])
    print("finished gmm")
    return super_pixels
