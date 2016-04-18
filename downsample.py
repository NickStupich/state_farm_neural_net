import cv2
import numpy as np

import sys
import os


classes = 10
resultSize = (64, 48)
pixels_length = resultSize[0] * resultSize[1]
data_length = pixels_length + 2

all_datas = []

for c in range(classes):
	folder = 'train/c%d' % c
	files = os.listdir(folder)

	class_data = np.zeros((len(files), data_length))

	for i, fn in enumerate(files):
		fn_num = int(fn.split('_')[1].split('.')[0])
		img = cv2.imread(folder + '/' + fn, 0)
		img_small = cv2.resize(img, resultSize)
		pixels = np.concatenate(img_small)

		data = np.concatenate(([c], [fn_num], pixels))

		class_data[i,:] = data

	all_datas.append(class_data)
	print('downsampled class %d' % c)

all_data = np.concatenate(all_datas)

print(all_data.shape)

np.save('downsampled_%s' % str(resultSize), all_data)