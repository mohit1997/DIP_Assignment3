import cv2
import numpy as np
from matplotlib import pyplot as plt

def highboostfilter(img, sigmaX, sigmaY, ksize, k=2):
	blur = cv2.GaussianBlur(img, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
	transform = np.uint8((k+1)*np.float32(img) - k*np.float32(blur))
	return transform

img = cv2.imread('sudoku.jpeg',0)
boostedfig = highboostfilter(img, sigmaX=1.0, sigmaY=1.0, ksize=(5, 5), k=1)


diagonal = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]]) 
kernelx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
laplaciand = cv2.filter2D(boostedfig, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)

crossx = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
crossy = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
robertcross = np.abs(cv2.filter2D(boostedfig, cv2.CV_8U, kernel=crossx)) + np.abs(cv2.filter2D(boostedfig, cv2.CV_8U, kernel=crossy))

sobel = np.abs(cv2.Sobel(boostedfig, cv2.CV_8U,1,0,ksize=3)) + np.abs(cv2.Sobel(img, cv2.CV_8U,0,1,ksize=3))


plt.figure(figsize=(6,6))
plt.subplot(2, 2, 1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2),plt.imshow(laplaciand,cmap = 'gray')
plt.title('LaplacianD'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3),plt.imshow(robertcross,cmap = 'gray')
plt.title('Robert Cross'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel 3x3'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("bestedge.pdf", format="pdf", dpi=200)