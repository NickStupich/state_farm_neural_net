import cv2
import numpy as np
import functools

def callback(x, y):
	print(x, y)

frame = np.ones((640, 640))

cv2.namedWindow('test')
cv2.createTrackbar('thrs0', 'test', 300, 800, functools.partial(callback, 0))
cv2.createTrackbar('thrs1', 'test', 300, 800, functools.partial(callback, 1))
# Do whatever you want with contours
cv2.imshow('test', frame)

cv2.waitKey(0)