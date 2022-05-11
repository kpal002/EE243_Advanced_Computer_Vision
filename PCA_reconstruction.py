from glob import glob
from PIL import Image
import numpy as np
from scipy import linalg
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# Calculate the R matrix

def calc_pca(img):

	mean = img.mean()

	img_zero_mean = np.subtract(img,mean)

	# np.cov calculates the covariance matrix X X^T
	matrix =  np.cov(img_zero_mean)
	u,s,vh = np.linalg.svd(matrix)

	

	# calculate the thresold in terms of variance
	total_var = np.sum(s)
	sum_eig=[]
	for i in range(len(s)):
		sum_eig.append(sum(s[:i])/total_var)


	# Plot thresold graph
	#plt.figure(figsize=(8, 6))
	#plt.plot(sum_eig)
	#plt.xlabel('No. of components')
	#plt.ylabel('variance')
	#plt.savefig('var.pdf')
	
	# Set the number of PCA components
	cutoff = 0
	for i in range(len(s)):
		if sum_eig[i]>= .98:
			cutoff = i
	
			break

	
	# Projection
	img_proj = img_zero_mean.T@u[:,0:cutoff]
	img_recons = img_proj@u[:,0:cutoff].T +mean
	
	return img_recons.T





if __name__ == '__main__':
	subj_img = []
	paths_files = glob("images/pca_face_images/*/*")
	for img in paths_files:
		image = Image.open(img)
		subj_img.append(image)

	recons_img = []
	for i in range(len(subj_img)):
		image_matrix  = np.asarray(subj_img[i])
		R = calc_pca(image_matrix)
		recons_img.append(R.ravel())

	
	recons_img_arr = np.asarray(recons_img)

	# clustering algorithm
	k_means = KMeans(init = "k-means++", n_clusters = 10, n_init=30,random_state=0,algorithm='elkan')
	k_means.fit(recons_img_arr)
	Z = k_means.predict(recons_img_arr)

	for i in range(0,10):
		row = np.where(Z==i)[0]  # row in Z for elements of cluster i
		num = row.shape[0]       #  number of elements for each cluster
		r = np.floor(num/10.)    # number of rows in the figure of the cluster 

		print("cluster "+str(i))
		print(str(num)+" elements")

		plt.figure(figsize=(10,10))

		for k in range(0, num):
			plt.subplot(r+1, 10, k+1)

			image = recons_img_arr[row[k], ]
			image = image.reshape(243, 320)
			plt.imshow(image, cmap='gray')
			plt.axis('off')

		plt.show()

	# Error estimation for a single image
	image2 = Image.open("images/pca_face_images/subject01/subject01_centerlight.png")
	image_matrix_err = np.asarray(image2)
	R_err = calc_pca(image_matrix_err)
	
	# Calculate the error between the original and the reconstructed image
	err = np.sum((image_matrix_err.astype("float") - R_err.astype("float")) ** 2)
	err /= float(R_err.shape[0] * R_err.shape[1])

	print(err)
	
	plt.show()

