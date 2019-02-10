import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sudoku.jpeg',0)

laplacian = cv2.Laplacian(img, cv2.CV_8U, scale=2)

diagonal = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]]) 
kernelx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])

laplacianX = cv2.filter2D(img, cv2.CV_8U, kernel=kernelx)
laplacianY = cv2.filter2D(img, cv2.CV_8U, kernel=kernelx.T)

laplacianXd = cv2.filter2D(img, cv2.CV_8U, kernel=diagonal+kernelx)
laplacianYd = cv2.filter2D(img, cv2.CV_8U, kernel=diagonal+kernelx.T)

laplaciand = cv2.filter2D(img, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)

sobel = np.abs(cv2.Sobel(img, cv2.CV_8U,1,0,ksize=3)) + np.abs(cv2.Sobel(img, cv2.CV_8U,0,1,ksize=3))

crossx = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
crossy = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])

robertcross = np.abs(cv2.filter2D(img, cv2.CV_8U, kernel=crossx)) + np.abs(cv2.filter2D(img, cv2.CV_8U, kernel=crossy))

plt.figure(figsize=(8,8))
plt.subplot(3, 3, 1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2),plt.imshow(laplacianX,cmap = 'gray')
plt.title('LaplacianX'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3),plt.imshow(laplacianY,cmap = 'gray')
plt.title('LaplacianY'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5),plt.imshow(laplacianXd,cmap = 'gray')
plt.title('LaplacianXD'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6),plt.imshow(laplacianYd,cmap = 'gray')
plt.title('LaplacianYD'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7),plt.imshow(laplaciand,cmap = 'gray')
plt.title('LaplacianD'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8),plt.imshow(robertcross,cmap = 'gray')
plt.title('Robert Cross'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("edge.pdf", format="pdf", dpi=200)