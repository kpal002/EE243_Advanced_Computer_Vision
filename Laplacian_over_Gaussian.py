import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Laplacian of the gaussian
def L_o_G(x, y, sigma):
    # Formatted this way for readability
    nom = ( (y**2)+(x**2)-2*(sigma**2) )
    denom = ( (2*np.pi*(sigma**6) ))
    expo = np.exp( -((x**2)+(y**2))/(2*(sigma**2)) )
    return nom*expo/denom






def create_log(sigma, size = 5):
    # Calculate w by using the size of filter and sigma value
    w = np.ceil(float(size)*float(sigma))

    #Check if it is even of not, if it even make it odd, by adding 1
    if(w%2 == 0):
        w = w + 1

    mask = []

    w_range = int(np.floor(w/2))
    #Iterate through the pixels and apply log filter and then append those changed pixels into a new array, and then reshape the array.

    for i in range(-w_range, w_range+1):
        for j in range(-w_range, w_range+1):
            mask.append(L_o_G(i,j,sigma))
    mask = np.array(mask)
    mask = mask.reshape(int(w),int(w))
    return mask


def convolve(image, mask):
    # Extract the heights and width of the image
    print(image.shape)
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



def zero_crossings(L_o_G_image):
    zero_image = np.zeros(L_o_G_image.shape)

    # Check the sign (negative or positive) of all the pixels around each pixel
    for i in range(1,L_o_G_image.shape[0]-1):
        for j in range(1,L_o_G_image.shape[1]-1):
            neg_count = 0
            pos_count = 0
            for a in range(-1, 2):
                for b in range(-1,2):
                    if(a != 0 and b != 0):
                        if(L_o_G_image[i+a,j+b] < 0):
                            neg_count += 1
                        elif(L_o_G_image[i+a,j+b] > 0):
                            pos_count += 1

            # If all the signs around the pixel are the same and they're not all zero, then it is not an edge. 
            # Otherwise, copy it to the edge map.
            zero = ( (neg_count > 0) and (pos_count > 0) )
            if(zero):
                zero_image[i,j] = 1

    return zero_image



if __name__ == '__main__':
    image = cv2.imread('peppers.tif')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create the L_o_G mask
    mask = create_log(2.5, 7.0)

    # Smooth the image by convolving with the LoG mask
    L_o_G_image = convolve(gray_image, mask)

    # Display the smoothed imgage
    fig = plt.figure(figsize=(8, 6))
    blurred = fig.add_subplot(121)
    blurred.imshow(L_o_G_image, cmap=cm.gray)

    # Find the zero crossings
    
    zero_image = zero_crossings(L_o_G_image)

    #Display the zero crossings
    edges = fig.add_subplot(122)
    edges.imshow(zero_image, cmap=cm.gray)
    plt.show()




