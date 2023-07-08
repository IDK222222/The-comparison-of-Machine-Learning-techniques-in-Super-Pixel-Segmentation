Superpixel Segmentation using GMM and Optimized K-means Clustering

This code is designed to perform superpixel segmentation using GMM and optimized K-means clustering.
The code also includes evaluation metrics to compare the segmentation results of both methods.

Prerequisites
    Before using this code, you need to install the following libraries:
        OpenCV (cv2)
        NumPy
        Matplotlib

How to use
Follow the steps below to use this code:
    Open "main.py" in any Python IDE or text editor.
    Modify line 10 input image name "img = cv2.imread('images/IMAGE_NAME.jpg')"
    Run the code.
    Define the bounding box around the object of interest using OpenCV's selectROI function.
    press space after defining the object.
    close all tabs that opened.
    The code will perform superpixel segmentation using GMM and optimized K-means clustering.
    It will generate segmentation results for both methods and save them in the "results" folder.
    It will also generate a text file named "Results.txt" in the "results" folder, which will contain the evaluation metrics for both methods.

How it works
The code performs the following steps:

Load the input image using OpenCV.
Define the bounding box around the object of interest using OpenCV's selectROI function.
Apply the GrabCut algorithm to generate a mask for the foreground.
Use the generated mask to get the cut_out image of the foreground.
Generate the ground truth image for evaluation purposes.
Perform superpixel segmentation using optimized K-means clustering.
Perform superpixel segmentation using GMM.
Compute evaluation metrics for both segmentation methods.
Save the segmentation results and evaluation metrics in the "results" folder.