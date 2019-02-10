import cv2
import numpy as np
from matplotlib import pyplot as plt

def highboostfilter(img, sigmaX, sigmaY, ksize, k=2):
	blur = cv2.GaussianBlur(img, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
	transform = np.uint8((k+1)*np.float32(img) - k*np.float32(blur))
	return transform

img = cv2.imread('sudoku.jpeg',0)

plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2),plt.imshow(highboostfilter(img, sigmaX=1.0, sigmaY=1.0, ksize=(5, 5)),cmap = 'gray')
plt.title('Highboost std=1.0 k=2'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3),plt.imshow(highboostfilter(img, sigmaX=5.0, sigmaY=5.0, ksize=(5, 5)),cmap = 'gray')
plt.title('Highboost std=5.0 k=2'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4),plt.imshow(highboostfilter(img, sigmaX=5.0, sigmaY=5.0, ksize=(5, 5), k=6),cmap = 'gray')
plt.title('Highboost std=5.0 k=5'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("highboost.pdf", format="pdf", dpi=500)