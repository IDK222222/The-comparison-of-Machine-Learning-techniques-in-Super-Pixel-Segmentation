import numpy as np
from skimage import segmentation


# Undersegmentation Error (UE):
def compute_ue(img, super_pixels, ground_truth):
    return 100 * np.sum(super_pixels != ground_truth) / img.size


# Achievable Segmentation Accuracy (ASA):
def compute_asa(img, super_pixels, ground_truth):
    k = len(np.unique(super_pixels))
    eps = 1e-10
    sigma = 0.5 * k / np.log(k) if k > 1 else 0
    asa = 0
    for i in range(k):
        mask = super_pixels == i
        area = np.sum(mask)
        if area == 0:
            continue
        overlap = np.sum(mask & ground_truth)
        union = np.sum(mask | ground_truth)
        asa += (area * (1 - np.exp(-(overlap + eps) / (sigma * area))) /
                (np.sqrt(area * union) + eps))
    return 100 * asa / img.size


# Boundary Recall (BR)
def compute_br(img, super_pixels, ground_truth, threshold=0.5):
    boundary = segmentation.find_boundaries(ground_truth)
    tp = np.sum(boundary & (super_pixels != ground_truth))
    fp = np.sum(boundary & (super_pixels == ground_truth))
    fn = np.sum(boundary & (ground_truth != 0) & (super_pixels != ground_truth))
    recall = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return 100 * f1


# compactness of super_pixels
def compute_superpixel_compactness(img, super_pixels):
    num_superpixels = np.unique(super_pixels).shape[0]
    compactness = np.zeros(num_superpixels)

    for i in range(num_superpixels):
        pixel_indices = np.where(super_pixels == i)
        pixel_values = img[pixel_indices]
        centroid = np.mean(pixel_values, axis=0)
        distances = np.linalg.norm(pixel_values - centroid, axis=1)
        avg_distance = np.mean(distances)
        compactness[i] = avg_distance

    overall_compactness = np.mean(compactness)
    return overall_compactness
