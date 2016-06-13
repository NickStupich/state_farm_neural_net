import numpy as np
import pylab
import cv2

import scipy.stats
# from pretrained_vgg16 import read_and_normalize_and_shuffle_train_data
from run_keras_cv_drivers_v2 import read_and_normalize_train_data


def display_img(img):
	if len(img.shape) == 3 and img.shape[0] == 3:
		pylab.imshow(np.transpose(img, (1, 2, 0)).astype(np.uint8))
	else:
		pylab.imshow(img)
	pylab.show()


train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(224, 224, 3, normalize=False)
threshold_value = 50


average_img_dict = {}
for driver in unique_drivers:
	print('running on driver: %s' % driver)
	indices = [i for i in range(len(driver_id)) if driver_id[i] == driver]
	print(len(indices))
	print(indices[:10])


	driver_imgs = train_data[indices]

	print('computing mean driver...')
	average_driver = np.mean(driver_imgs, axis=0)
	std_dev_driver = np.sum(np.std(driver_imgs, axis=0), axis=0)

	print(average_driver.shape)

	for img in driver_imgs[::50]:

		disp_num = 1
		pylab.subplot(3, 3, disp_num)
		disp_num += 1
		pylab.imshow(np.transpose(average_driver, (1, 2, 0)).astype(np.uint8))
		pylab.title('average driver')


		pylab.subplot(3, 3, disp_num)
		disp_num += 1
		pylab.imshow(std_dev_driver.astype(np.uint8))
		pylab.title('stddev driver')


		pylab.subplot(3, 3, disp_num)
		disp_num += 1
		pylab.imshow(np.transpose(img, (1, 2, 0)).astype(np.uint8))
		pylab.title('test driver')

		diffs = np.abs(np.sum(img - average_driver, axis=0))
		print(diffs.shape)

		print(diffs)

		pylab.subplot(3, 3, disp_num)
		disp_num += 1
		pylab.imshow(diffs.astype(np.uint8))
		pylab.title('diffs')

		pos_indices = np.where(diffs > threshold_value)

		mask = np.zeros(diffs.shape, dtype=np.uint8)
		mask[pos_indices] = 1


		#ret, mask = cv2.threshold(diffs, threshold_value, 1, cv2.THRESH_BINARY)
		#print(ret)
		#print(mask[:100])

		pylab.subplot(3, 3, disp_num)
		disp_num += 1
		pylab.imshow(mask.astype(np.uint8))
		pylab.title('mask')


		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations = 1)
		mask = cv2.erode(mask, kernel, iterations = 1)
		mask = cv2.erode(mask, kernel, iterations = 1)
		mask = cv2.dilate(mask, kernel, iterations = 1)

		pylab.subplot(3, 3, disp_num)
		disp_num += 1
		pylab.imshow(mask.astype(np.uint8))
		pylab.title('mask filtered')

		# mask = scipy.stats.threshold(diffs, threshold_value)
		# print(diffs[:100])

		# mask = np.clip(mask, 0, threshold_value) / threshold_value
		# print(diffs[:100])


		# display_img(mask)


		masked_img = mask*img
		print(masked_img.shape)

		pylab.subplot(3, 3, disp_num)
		disp_num += 1
		pylab.imshow(np.transpose(masked_img, (1, 2, 0)).astype(np.uint8))
		pylab.title('masked')


		pylab.show()


	break