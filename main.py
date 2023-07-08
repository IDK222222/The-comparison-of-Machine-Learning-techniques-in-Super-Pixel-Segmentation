import cv2
import gmm
import kmeans
import time
import numpy as np
import matplotlib.pyplot as plt
from evaluating_metrics import compute_ue, compute_asa, compute_br, compute_superpixel_compactness

# Load the image
img = cv2.imread('images/butterfly1.jpg')

# Define the bounding box around the object of interest
bbox = cv2.selectROI(img)
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Apply the GrabCut algorithm
cv2.grabCut(img, mask, bbox, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Generate the mask for the foreground
mask2 = np.where((mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 255, 0).astype('uint8')
fgd_mask = cv2.grabCut(img, None, bbox, None, None, 5, cv2.GC_INIT_WITH_RECT)[0]

# Apply the mask to the original image to get the cut_out
cut_out = cv2.bitwise_and(img, img, mask=mask2)

# Generate the ground truth image
ground_truth = np.zeros_like(fgd_mask)
ground_truth[fgd_mask == cv2.GC_PR_FGD] = 1  # set probable foreground pixels to 1
ground_truth[fgd_mask == cv2.GC_FGD] = 2  # set definite foreground pixels to 2


# Display the cut_out and the combined image
cv2.imshow("Cut out", cut_out)
cv2.imshow("Ground truth", ground_truth * 127)

# Save the images
cv2.imwrite("results/cut_out.png", cut_out)
cv2.imwrite("results/ground_truth.png", ground_truth * 127)
cv2.waitKey(0)

# Convert the image to a NumPy array
img = np.array(img)
# Check for NaN or Inf values and replace them with 0
img[np.isnan(img)] = 0
img[np.isinf(img)] = 0

# starting a timer
kmeans_start_time = time.time()

# Perform super pixel segmentation using optimized K-means clustering
super_pixels_kmeans = kmeans.kmeans_super_pixel(img, n_clusters=10)

#ending the timer
kmeans_end_time = time.time()

#gmm start time
gmm_start_time = time.time()

# Perform super pixel segmentation using GMM
super_pixels_gmm = gmm.gmm_super_pixel(img,10)

#gmm_end_time
gmm_end_time = time.time()

# Compute the evaluation metrics for GMM segmentation
ue_gmm = compute_ue(img, super_pixels_gmm, ground_truth)
asa_gmm = compute_asa(img, super_pixels_gmm, ground_truth)
br_gmm = compute_br(img, super_pixels_gmm, ground_truth)
compactness_gmm = compute_superpixel_compactness(img, super_pixels_gmm)
time_gmm = gmm_end_time - gmm_start_time

# Compute the evaluation metrics for optimized k-means segmentation
ue_kmeans = compute_ue(img, super_pixels_kmeans, ground_truth)
asa_kmeans = compute_asa(img, super_pixels_kmeans, ground_truth)
br_kmeans = compute_br(img, super_pixels_kmeans, ground_truth)
compactness_kmeans = compute_superpixel_compactness(img,super_pixels_kmeans)
time_kmeans = kmeans_end_time - kmeans_start_time

# Print the results and save to a file
with open("results/Results.txt", "w") as file:
    file.write("GMM segmentation results:\n")
    file.write("UE: {}\n".format(ue_gmm))
    file.write("ASA: {}\n".format(asa_gmm))
    file.write("BR: {}\n".format(br_gmm))
    file.write("Compactness: {}\n".format(compactness_gmm))
    file.write("Time: {}\n".format(time_gmm))

    file.write("\nOptimized k-means segmentation results:\n")
    file.write("UE: {}\n".format(ue_kmeans))
    file.write("ASA: {}\n".format(asa_kmeans))
    file.write("BR: {}\n".format(br_kmeans))
    file.write("Compactness: {}\n".format(compactness_kmeans))
    file.write("Time: {}\n".format(time_kmeans))

# Print the results
print("GMM segmentation results:")
print("UE: ", ue_gmm)
print("ASA: ", asa_gmm)
print("BR: ", br_gmm)
print("Compactness: ",compactness_gmm)
print("Time:",time_gmm)

print("Optimized k-means segmentation results:")
print("UE: ", ue_kmeans)
print("ASA: ", asa_kmeans)
print("BR: ", br_kmeans)
print("Compactness: ",compactness_kmeans)
print("Time:",time_kmeans)

# Load the original image and the segmented image
original_image = img  # Load the original image
segmented_image1 = super_pixels_kmeans # Load the K-means segmented image
segmented_image2 = super_pixels_gmm # Load the GMM segmented image

# Normalize the images
segmented_image1_normalized = cv2.normalize(segmented_image1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
segmented_image2_normalized = cv2.normalize(segmented_image2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Convert color format from RGB to BGR for OpenCV
segmented_image1_bgr = cv2.cvtColor(segmented_image1_normalized, cv2.COLOR_RGB2BGR)
segmented_image2_bgr = cv2.cvtColor(segmented_image2_normalized, cv2.COLOR_RGB2BGR)

cv2.imwrite('results/kmeans.jpg', segmented_image1_bgr)
cv2.imwrite('results/gmm.jpg', segmented_image2_bgr)

# Plot the original and segmented images
plt.figure()
plt.subplot(1,5,1)
plt.imshow(original_image)
plt.title("Original")

plt.subplot(1,5,2)
plt.imshow(ground_truth)
plt.title("Ground Truth")

plt.subplot(1,5,3)
plt.imshow(cut_out)
plt.title("Cut_out")

plt.subplot(1,5,4)
plt.imshow(segmented_image2)
plt.title("GMM")

plt.subplot(1,5,5)
plt.imshow(segmented_image1)
plt.title("Kmeans")

plt.savefig("results/Results")
plt.show()

