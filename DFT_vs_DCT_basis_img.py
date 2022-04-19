# Import functions and libraries
import numpy as np
import cv2
from scipy.fftpack import fft, dct
import matplotlib.cm as cm
import matplotlib.pyplot as plt




# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

image = cv2.imread("gonzalezwoods725.PNG")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imsize = gray_image.shape

M = 4
F_1 = np.zeros((4,4))
F_2 = np.zeros((4,4),dtype = complex)

for k in range(4):
    if k == 0:
        alpha = np.sqrt(1/M)
    else:
        alpha = np.sqrt(2/M)
    for n in range(4):
        F_1[n,k] = alpha*np.cos(k*np.pi*(2*n+1)/(2*M))
        F_2[n,k] = np.exp(-1j*2*np.pi*k*n/M)

# PLOT THE 4X4 DCT basis
# define subplot grid

plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("DCT basis functions", fontsize=18, y=0.95)

k = 1
for p in range(4):
	for q in range(4):
		dct_basis = np.outer(F_1[:,p],F_1[:,q])
		plt.subplot(4,4,k)
		plt.imshow(dct_basis, cmap=cm.gray, extent=[0, 1, 0, 1])
		k = k+1

#plt.show()

# PLOT THE 4X4 DFT basis
plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("DFT basis functions", fontsize=18, y=0.95)

k = 1
for u in range(4):
	for v in range(4):
		dft_basis = np.outer(F_2[:,u],F_2[:,v])
		dft_basis_r = dft_basis.real - dft_basis.imag
		plt.subplot(4,4,k)
		plt.imshow(dft_basis_r, cmap=cm.gray)
		k = k+1

plt.show()



J = np.fft.fft2(gray_image)
J_shifted = np.fft.fftshift(J)
fig = plt.figure(figsize=(12, 10))
fig.subplots_adjust(hspace=0.5)
fig.suptitle("DFT and DCT transformed images", fontsize=18, y=0.95)
ax1 = plt.subplot(2,2,1)
ax1.imshow(10*np.log10(np.abs(J_shifted)))
ax1.title.set_text('Magnitude of the DFT')

ax2 = plt.subplot(2,2,2)
ax2.imshow(np.angle(J),extent=[0, 1, 0, 1])
ax2.title.set_text('Phase of the DFT')


L = dct2(gray_image)
ax3 = plt.subplot(2,2,3)
ax3.imshow(L[0:16,0:16])
ax3.title.set_text('Small section of DCT transformed image');
plt.show()

