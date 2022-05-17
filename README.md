# EE243_Advanced_Computer_Vision
Graduate course in Computer Vision in Spring'22 at UC Riverside

I am taking graduate Computer Vision course during Spring'22. This repository will contain all the assignments, final course project and other important algorithms implemented throughout the quarter.

# Contents

1. Edge detection \
  a. Laplacian filter\
  b. Canny edge detector\
  c. Hough transform
  
2. Fourier transform and basis images.
3. Shi-Tomasi corner detector with the following features\
  a. Sobel operator to obtain the gradients.\
  b. Window size of 5 X 5 \
  c. Suppress points which are not local maxima within a 5 × 5 window.\
  
4. Feature extraction using CNN.
5. Image reconstruction using PCA annalysis.
6. Normalized cuts for image segmentation.
7. Expectation-Maximization algorithm for mixture of Gaussian model based on color features for image segmentation.
8. Given a recorded video from a static over-head camera that contains people walking in a lobby. The task is to detect the persons, extract features for the detected regions and then apply data association to obtain correspondence between persons from frame t to t + 1. \
  a. It first loads the video and extracts the frames from it. For each frame, it calls the following functions. \
  b. As the video is recorded from a static camera, a simple sum of difference between N frames will be able to highlight the moving objects. The getSumOfDiff.m function is called for this purpose with N = 3. \
  c. The sum of difference image is then used by the getDetections.m function to obtain the detections. This function should segment the blobs which are highlighted in the sum of difference image and return the bounding box details [topleft x,topleft y,width,height] of the segmented blobs. \
  d. The features are extracted from the bounding box regions using the HoG features implemented in getFeatures.m function. \
  e. All the above steps are repeated for the next frame. Then, the getMatches.m code is used to obtain the correspondences between the detected regions in the two frames. \
Detailed code can be found here [Link](https://github.com/username/repoName/somePathTo/myExampleCode)

# In progress
