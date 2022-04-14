import numpy as np 
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt




#########################################################################
# The algorithm runs in 5 separate steps:
# 1. Smoothing: Blurring of the image to remove noise.
# 2. Finding gradients: The edges should be marked where the gradients of the image has large magnitudes.
# 3. Non-maximum suppression: Only local maxima should be marked as edges.
# 4. Double thresholding: Potential edges are determined by thresholding.
# 5. Edge tracking by hysteresis: Final edges are determined by suppressing
#    all edges that are not connected to a very certain (strong) edge.
##########################################################################

	   
def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def convolve(image, mask):
    # Extract the heights and width of the image
    
    width = image.shape[1]
    height = image.shape[0]

    # make a range of pixels which are covered by the mask (output of filter)
    w_range = int(np.floor(mask.shape[0]/2))

    # Make an array of zeros (res_image)
    res_image = np.zeros((height, width))

    # Iterate over every pixel that can be covered by the mask
    for i in range(w_range,width-w_range):
        for j in range(w_range,height-w_range):
            # Then convolute with the mask 
            for k in range(-w_range,w_range+1):
                for h in range(-w_range,w_range+1):
                    res_image[j, i] += mask[w_range+h,w_range+k]*image[j+h,i+k]

    return res_image


def sobel_filters(img):
    # Taken from https://en.wikipedia.org/wiki/Sobel_operator

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)


def non_max_suppression(img, D):
    M, N = img.shape

    # Create a matrix initialized to 0 of the same size of the original gradient intensity matrix
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0
    
    return Z

# Double threshold
def threshold(img, highThreshold, lowThreshold):
    
    M, N = img.shape


    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res



def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img



if __name__ == '__main__':
    image = cv2.imread('peppers.tif')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_smoothed = convolve(gray_image,gaussian_kernel(size=3,sigma=1.0))
    fig = plt.figure()
    image1 = fig.add_subplot(1,4,1)
    image1.imshow(img_smoothed, cmap=cm.gray)
    image1.title.set_text('Gaussian smoothed image')

    gradient_mag, gradient_dir = sobel_filters(img_smoothed)
    nonMaxImg = non_max_suppression(gradient_mag, gradient_dir)
    image2 = fig.add_subplot(1,4,2)
    image2.imshow(nonMaxImg, cmap=cm.gray)
    image2.title.set_text('Non-max Suppression')

    thresholdImg = threshold(nonMaxImg,20,5)
    image3 = fig.add_subplot(1,4,3)
    image3.imshow(thresholdImg, cmap=cm.gray)
    image3.title.set_text('Thresholding')

    img_final = hysteresis(thresholdImg, weak=25)
    image4 = fig.add_subplot(1,4,4)
    image4.imshow(img_final, cmap=cm.gray)
    image4.title.set_text('Final image after hysteresis')
    plt.show()
