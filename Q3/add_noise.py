import cv2
import numpy as np
import matplotlib.pyplot as plt

def addsnp(img, prob):
	temp = img.copy()

	matsnp = np.random.rand(temp.shape[0], temp.shape[1])

	temp[matsnp < prob] = 0
	temp[matsnp > 1-prob] = 255

	return temp

def harmonic_filter(img, ksize=(5, 5)):
	temp = np.float32(img.copy())
	temp = 1.0/(temp+1e-8)
	blur = cv2.blur(temp,ksize)
	blur = np.uint8(1.0/blur)
	return blur

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

noise_img = addsnp(img, 0.1)



plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2),plt.imshow(noise_img, cmap = 'gray')
plt.title('Salt & Pepper Prob=0.05'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3),plt.imshow(harmonic_filter(noise_img, ksize=(3, 3)), cmap = 'gray')
plt.title('Harmonic Filter Kernel=(3,3)'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4),plt.imshow(cv2.medianBlur(noise_img,3), cmap = 'gray')
plt.title('Median k=3'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5),plt.imshow(adaptivemedian(noise_img,3), cmap = 'gray')
plt.title('Adaptive Median k={3, 5, 7}'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("filternoise.pdf", format="pdf", dpi=200)
