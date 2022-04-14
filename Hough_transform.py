import numpy as np 
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.measure import LineModelND, ransac
from Canny_edge_detector import gaussian_kernel, convolve, sobel_filters, non_max_suppression, threshold, hysteresis




img = cv2.imread('house.tif')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny edge detection part
img_smoothed = convolve(gray_image,gaussian_kernel(size=5,sigma=2.0))
gradient_mag, gradient_dir = sobel_filters(img_smoothed)
nonMaxImg = non_max_suppression(gradient_mag, gradient_dir)
thresholdImg = threshold(nonMaxImg, 20, 10)
img_final = hysteresis(thresholdImg, weak=25)


# Resolution of the output lines
distance_resolution = 2
angular_resolution = np.pi/180.0 / 6

height, width = img_final.shape

# the maximum possible line length
max_rho = np.sqrt(width**2 + height**2)

# the number of linear steps (+/- max rho)
num_rho = int(2 * max_rho / distance_resolution) - 1
max_rho = num_rho * distance_resolution / 2

# the number of discrete angular steps
num_theta = int(np.pi / angular_resolution)


accumulator = np.zeros((num_rho, num_theta)).astype(np.int32)

def calc_y(x, rho, theta):
    if theta == 0:
        # avoid divide by 0 in the slope
        return rho
    else:
        return (-np.cos(theta) / np.sin(theta)) * x + (rho / np.sin(theta))


def polar_to_xy(rho, theta):
    x1 = 0
    x2 = width
    y1 = int(calc_y(0, rho, theta))
    y2 = int(calc_y(width, rho, theta))
    return (x1, y1), (x2, y2)


edge_points= [] 
for x in range(0, width):
    for y in range(0, height):
        
        # edge colour - will be 255 (white)
        edge_colour = img_final[y,x]

        # if there's an edge
        if edge_colour != 0:
            
            edge_points.append((x,y))

            # array of each of the discrete values of theta
            theta_arr = np.arange(0, np.pi, angular_resolution)

            # Line equation in polar coordinates
            rho_arr = x * np.cos(theta_arr) + y * np.sin(theta_arr)

            
            # Calculate the locus of points lying approximately on a sinusoidal
            # curve for each edge point (x, y)
            rho_index_arr = np.rint(rho_arr/distance_resolution + num_rho/2).astype(np.int32)
            theta_index_arr = np.rint(theta_arr/angular_resolution).astype(np.int32)
            
            acc = np.bincount(rho_index_arr * accumulator.shape[1] + theta_index_arr)
            acc.resize(accumulator.shape)
            
            accumulator += acc


# To be used for RANSAC algorithms
edge_points = np.array(edge_points)


# Try a different colour map
plt.figure(figsize=(16, 18))
plt.imshow(accumulator, cmap='inferno')
plt.xlabel('theta')
plt.ylabel('rho')
plt.show()



flat = accumulator.flatten()
plt.figure()
plt.hist(flat[flat > 0], bins=250, log=True)
plt.show()


# Chosen from the histogram
threshold = 140
thresholded_accumulator = accumulator.copy()
thresholded_accumulator[thresholded_accumulator <= threshold ] = 0 


# Trial and error for choosing the optimal distance
min_distance = 4
coordinates = peak_local_max(thresholded_accumulator, min_distance=min_distance)




# circle the local peaks / maxima
fig = plt.figure(figsize=(16, 18))
plt.imshow(accumulator, cmap='viridis')
plt.axis('off')
plt.scatter(coordinates[:,1], coordinates[:,0], 30, color='w', marker='.')
extent = fig.gca().get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('accumulator_points.png', bbox_inches=extent, pad_inches=0)



# Now convert all these points back to lines,
# and draw lines on a copy of the image
image_output = img.copy()

red = (0, 0, 255) # The color chosen for the lines.

for rho_index, theta_index in coordinates:
    rho = (rho_index - num_rho/2) * distance_resolution
    theta = theta_index * angular_resolution
#   print((rho_index, theta_index), (rho, theta))
    (x1, y1), (x2, y2) = polar_to_xy(rho, theta)

    cv2.line(image_output, (x1, y1), (x2, y2), red, 1, cv2.LINE_AA)

# 50% "transparency" lines
image_output = cv2.addWeighted(img.copy(), 0.5, image_output, 0.5, 1)

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR)) 
cv2.imwrite('output.png', image_output)




