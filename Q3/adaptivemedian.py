import cv2
import numpy as np
import matplotlib.pyplot as plt

# from numpy import *

def addsnp(img, prob):
	temp = img.copy()

	matsnp = np.random.rand(temp.shape[0], temp.shape[1])

	temp[matsnp < prob] = 0
	temp[matsnp > 1-prob] = 255

	return temp

def demo(target_array):
	return int(np.median(target_array)), int(np.max(target_array)), int(np.min(target_array))

def adaptivemedian(image, window=3):

	## set filter window and image dimensions
	W = 2*window + 1
	xlength, ylength = img.shape

	vlength = W*W

	## create 2-D image array and initialize window
	paddedimage = np.pad(image.copy(), window, 'constant', constant_values=255)
	print(np.max(image))
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
print(img.shape)
noise_img = addsnp(img, prob=0.1)
output = adaptivemedian(noise_img)

print(output.shape)
plt.imshow(output, cmap='gray')
plt.show()