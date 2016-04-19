import cv2
import numpy as np

import sys
import os


classes = 10
resultSize = (64, 48)
isColor = 1
pixels_length = resultSize[0] * resultSize[1]
data_length = pixels_length * (3 if isColor else 1) + 2

all_datas = []

debug = False

if debug:
	cv2.namedWindow('full', cv2.WINDOW_NORMAL)
	cv2.namedWindow('small', cv2.WINDOW_NORMAL)

for c in range(classes):
	folder = 'train/c%d' % c
	files = os.listdir(folder)

	class_data = np.zeros((len(files), data_length))

	for i, fn in enumerate(files):
		fn_num = int(fn.split('_')[1].split('.')[0])
		img = cv2.imread(folder + '/' + fn, isColor)

		if debug:
			cv2.imshow("full", img)
			cv2.waitKey(0)

		img_small = cv2.resize(img, resultSize, interpolation = cv2.INTER_AREA)

		pixels = np.concatenate(np.concatenate(img_small, axis=1))

		if debug:
			print(pixels.shape)
			cv2.imshow("small", img_small)
			cv2.waitKey(0)

		data = np.concatenate(([c], [fn_num], pixels))

		class_data[i,:] = data

	all_datas.append(class_data)
	print('downsampled class %d' % c)

all_data = np.concatenate(all_datas)

print(all_data.shape)

np.save('downsampled_%s_%d' % (str(resultSize), isColor), all_data)