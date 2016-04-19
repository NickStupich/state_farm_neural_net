import cv2
import numpy as np

import pylab

import sys
import os

classes = 1
isColor = 0

i=0
all_files = []

for c in range(classes):
	folder = 'train/c%d' % c
	files = os.listdir(folder)
	all_files += list(map(lambda s: folder + '/' + s, files))



all_data = np.zeros((len(all_files), 640*480 * (3 if isColor else 1)), dtype='uint8')

for i, fn in enumerate(all_files):
	img = cv2.imread(fn, isColor)

	all_data[i] = np.ndarray.flatten(img)

	if i % 100 == 0:
		print i, len(all_files)

print 'loaded all data'

pairs = []
for i in range(len(all_files)):
	if i % 1 == 0:
		print i
	for j in range(i+1, len(all_files)):
		d = np.mean(np.abs(all_data[i] - all_data[j]))
		pairs.append((i, j, d))

print 'got all pairs'

distances = np.array(map(lambda x: x[2], pairs))

pylab.hist(distances); pylab.show()


# for i in range(len(filese))