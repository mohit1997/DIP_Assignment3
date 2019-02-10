import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sudoku.jpeg',0)

sobel3 = np.abs(cv2.Sobel(img, cv2.CV_8U,1,0,ksize=3)) + np.abs(cv2.Sobel(img, cv2.CV_8U,0,1,ksize=3))
sobel5 = np.abs(cv2.Sobel(img, cv2.CV_8U,1,0,ksize=5)) + np.abs(cv2.Sobel(img, cv2.CV_8U,0,1,ksize=5))
sobel7 = np.abs(cv2.Sobel(img, cv2.CV_8U,1,0,ksize=7)) + np.abs(cv2.Sobel(img, cv2.CV_8U,0,1,ksize=7))


plt.figure(figsize=(6,6))
plt.subplot(2, 2, 1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2),plt.imshow(sobel3,cmap = 'gray')
plt.title('Sobel 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3),plt.imshow(sobel5,cmap = 'gray')
plt.title('Sobel 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4),plt.imshow(sobel7,cmap = 'gray')
plt.title('Sobel 7x7'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("sobelvariation.pdf", format="pdf", dpi=200)