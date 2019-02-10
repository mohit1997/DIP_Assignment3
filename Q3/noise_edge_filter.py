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

def demo(target_array):
	return int(np.median(target_array)), int(np.max(target_array)), int(np.min(target_array))

def adaptivemedian(image, window=3):

	## set filter window and image dimensions
	W = 2*window + 1
	xlength, ylength = img.shape

	vlength = W*W

	## create 2-D image array and initialize window
	paddedimage = np.pad(image.copy(), window, 'constant', constant_values=255)
	image_array = np.array(paddedimage, dtype=np.uint8)
	filter_window = np.array(np.zeros((W,W)))
	target_vector = np.array(np.zeros(vlength))
	changed = 0

	try:
		for y in range(window, ylength+window):
			for x in range(window, xlength+window):
				for w in range(1, window):
					filter_window = image_array[x-w:x+w+1, y-w:y+w+1]
					# print(filter_window)
					# target_vector = np.reshape(filter_window, ((vlength),))
					median, max_pixel, min_pixel = demo(filter_window)
					if median < max_pixel and median > min_pixel:
						if paddedimage[x, y] == min_pixel or paddedimage[x, y] == max_pixel:
							image_array[x, y] = median
							changed += 1
						break
	except TypeError:
		print "Error in processing function:", err
		sys.exit(2)
	print changed, "pixel(s) filtered out of", xlength*ylength
	
	## return only central array
	return image_array[window:-window, window:-window]

img = cv2.imread('sudoku.jpeg',0)

noise_img = addsnp(img, 0.05)

n0 = addsnp(img, 0.1)
n10 = addsnp(img, 0.01)
n30 = addsnp(img, 0.001)
n60 = addsnp(img, 0.0001)

f0 = adaptivemedian(n0)
f10 = adaptivemedian(n10)
f30 = adaptivemedian(n30)
f60 = adaptivemedian(n60)

boosted0 = highboostfilter(f0, sigmaX=1.0, sigmaY=1.0, ksize=(5, 5), k=1)
boosted10 = highboostfilter(f0, sigmaX=1.0, sigmaY=1.0, ksize=(5, 5), k=1)
boosted30 = highboostfilter(f30, sigmaX=1.0, sigmaY=1.0, ksize=(5, 5), k=1)
boosted60 = highboostfilter(f60, sigmaX=1.0, sigmaY=1.0, ksize=(5, 5), k=1)

diagonal = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]]) 
kernelx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])


sobel0 = cv2.filter2D(boosted0, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)
sobel10 = cv2.filter2D(boosted10, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)
sobel30 = cv2.filter2D(boosted30, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)
sobel60 = cv2.filter2D(boosted60, cv2.CV_8U, kernel=diagonal+kernelx+kernelx.T)


plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1),plt.imshow(sobel0, cmap = 'gray')
plt.title('SNR=0dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2),plt.imshow(sobel10,cmap = 'gray')
plt.title('SNR=10dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3),plt.imshow(sobel30, cmap = 'gray')
plt.title('SNR=30dB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4),plt.imshow(sobel60,cmap = 'gray')
plt.title('SNR=60dB'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("bestfilterededge.pdf", format="pdf", dpi=200)