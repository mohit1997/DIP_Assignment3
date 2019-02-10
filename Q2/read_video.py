import cv2
import numpy as np


if __name__ == "__main__":
    filename = "man.mp4"
    # size = (360, 640)
    cap = cv2.VideoCapture(filename)
    i = 1
    while 1:
        ret, frame = cap.read()
        if ret:
            print(frame.shape)
            cv2.imshow("frame", frame)
            cv2.imwrite("frames/frame-" + str(i) + ".jpg", frame)
            if (i>1):
                # Gradient Across Image
                edges = np.uint8(np.abs(np.float32(frame) - np.float32(prevframe)))
                # Combining gradient across channels
                edge = np.sum(edges, axis=-1)
                # Thresholding
                edge[edge<50] = 0
                edge[edge>0] = 255
                
                cv2.imwrite("edges/frame-" + str(i) + ".jpg", edge)
            i = i+1
            cv2.waitKey(30)
            prevframe = frame
        else:
            break