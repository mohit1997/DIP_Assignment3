import cv2
import numpy as np
import matplotlib.pyplot as plt

def addsnp(img, prob):
	temp = img.copy()

	matsnp = np.random.rand(temp.shape[0], temp.shape[1])

	temp[matsnp < prob] = 0
	temp[matsnp > 1-prob] = 255

	return temp

def highboostfilter(img, sigmaX, sigmaY, ksize, k=2):
	blur = cv2.GaussianBlur(img, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
	transform = np.uint8((k+1)*np.float32(img) - k*np.float32(blur))
	return transform

img = cv2.imread('sudoku.jpeg',0)

noise_img = addsnp(img, 0.05)

n0 = addsnp(img, 0.1)
n10 = addsnp(img, 0.01)
n30 = addsnp(img, 0.001)
n60 = addsnp(img, 0.0001)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2),plt.imshow(n0,cmap = 'gray')
plt.title('Salt&Pepper SNR=0dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3),plt.imshow(n10, cmap = 'gray')
plt.title('Salt&Pepper SNR=10dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4),plt.imshow(n30,cmap = 'gray')
plt.title('Salt&Pepper SNR=30dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5),plt.imshow(n60,cmap = 'gray')
plt.title('Salt&Pepper SNR=60dB'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("noise.pdf", format="pdf", dpi=200)

diagonal = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]]) 
kernelx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
# laplaciand = cv2.filter2D(boostedfig, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)

sobel0 = cv2.filter2D(n0, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)
sobel10 = cv2.filter2D(n10, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)
sobel30 = cv2.filter2D(n30, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)
sobel60 = cv2.filter2D(n60, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)


plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2),plt.imshow(sobel0,cmap = 'gray')
plt.title('Sobel SNR=0dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3),plt.imshow(sobel10, cmap = 'gray')
plt.title('Sobel SNR=10dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4),plt.imshow(sobel30,cmap = 'gray')
plt.title('Sobel SNR=30dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5),plt.imshow(sobel60,cmap = 'gray')
plt.title('Sobel SNR=60dB'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("rawedgelaplacian.pdf", format="pdf", dpi=200)