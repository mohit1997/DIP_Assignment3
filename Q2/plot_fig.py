import numpy as np
import matplotlib.pyplot as plt
import cv2

edge_dir = "edges/frame-"
frame_dir = "frames/frame-"

plt.figure(figsize=(15, 10))

for i in range(1, 5*5+1):
	frame = cv2.imread(frame_dir+str(i)+'.jpg', 1)
	plt.subplot(5, 5, i),plt.imshow(frame, cmap = 'gray')
	plt.title("Frame "+ str(i)), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig('framefig.pdf')

plt.figure(figsize=(15, 10))

for i in range(1, 5*5+1):
	frame = cv2.imread(edge_dir+str(i+1)+'.jpg', 1)
	plt.subplot(5, 5, i),plt.imshow(frame, cmap = 'gray')
	plt.title("Difference "+ str(i)), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig('edgefig.pdf', dpi=500)
