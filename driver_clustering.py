import numpy as np
import pylab
import cv2

# from pretrained_vgg16 import read_and_normalize_and_shuffle_train_data
from run_keras_cv_drivers_v2 import read_and_normalize_train_data

from sklearn.cluster import KMeans

# img_shape = (224, 224, 3)

# img_shape = (48, 64, 3)

# train_data, train_target, driver_id, unique_drivers = read_and_normalize_and_shuffle_train_data(*img_shape)
train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(224, 224, 3, normalize=False)

n = 5000
train_data = train_data[:n]
driver_id = driver_id[:n]

box_size = 30
#np.arange(box_size, 224-box_size)
#allow = np.concatenate((np.arange(box_size), np.arange(224-box_size, 224)))
# train_data[:, :, box_size:-box_size, box_size:-box_size] = 0
# train_data = train_data[:, :, allow, allow]
#train_data = train_data[:, :, -30:, -30:]

flat_data = np.reshape(train_data, (-1, np.prod(train_data.shape[1:])))
train_data = None

print(flat_data.shape)
print(flat_data.dtype)
# print(train_data.shape)

kmeans = KMeans(n_clusters = len(unique_drivers), 
				# max_iter = 10, 
				# n_init=1, 
				precompute_distances=True, 
				n_jobs=-2)

id_dict = dict([(driver, i) for i, driver in enumerate(unique_drivers)])

print('starting fit...')
predictions = kmeans.fit_predict(flat_data)
print('done fitting and predicting')

confusion_mat = np.zeros((len(unique_drivers), len(unique_drivers)))

for pred, real_str in zip(predictions, driver_id):
	real_id = id_dict[real_str]
	confusion_mat[real_id, pred] += 1

print(np.where(np.ndarray.flatten(confusion_mat) > 0)[0].shape)
print('num unique drivers: %d' % len(unique_drivers))

#print('\n'.join(map(str, zip(predictions, driver_id))))
np.set_printoptions(threshold=np.nan)
print(confusion_mat)

print('confusion max diagonals: %d' % np.sum(np.amax(confusion_mat, axis=0)))
print('confusion max diagonals: %d' % np.sum(np.amax(confusion_mat, axis=1)))

centers = kmeans.cluster_centers_
np.save('cluster_centers.npy', centers)


center_imgs = np.reshape(centers, (-1, 3, 224, 224))

for center_img in center_imgs: #np.reshape(flat_data, (-1, 3, 224, 224)):
	# pylab.imshow(cv2.cvtColor(np.transpose(center_img, (1, 2, 0)), cv2.COLOR_BGR2RGB))
	pylab.imshow(np.transpose(center_img, (1, 2, 0)).astype(np.uint8))#[:, :, [1, 1, 1]])
	pylab.show()

	# print(center_img.dtype)
	# print(np.mean(center_img))
	# cv2.imshow('img', np.transpose(center_img, (1, 2, 0)))
	# cv2.waitKey(0)

