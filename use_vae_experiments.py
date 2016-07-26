import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import run_keras_cv_drivers_v2

def cluster_and_classify(encoder, all_data):
	x_train, y_train, train_id, driver_id, unique_drivers = run_keras_cv_drivers_v2.read_and_normalize_train_data(img_rows, img_cols, color_type)
	driver_id_int = list(map(lambda x: int(x.strip('p')), driver_id))
	total_driver_counts = np.zeros(100, dtype='int')
	for i in driver_id_int: total_driver_counts[i] += 1
	to_encode = x_train
	#to_encode = all_data


	# display a 2D plot of the digit classes in the latent space
	encoded_data = encoder.predict(to_encode, batch_size=batch_size)
	#labels = np.array(list(map(lambda id_str: int(id_str.strip('p')), driver_id)))
	#class_labels = np.argmax(y_train, axis=1)
	num_clusters = 10
	kmeans = KMeans(n_clusters = num_clusters)
	pred_labels = kmeans.fit_predict(encoded_data)

	print(pred_labels[:100])

	if 0:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		n = 5000
		skip = len(encoded_data) / n
		#ax.scatter(encoded_data[::skip, 0], encoded_data[::skip, 1], zs=encoded_data[::skip, 2], c=labels[::skip])
		ax.scatter(encoded_data[::skip, 0], encoded_data[::skip, 1], zs=encoded_data[::skip, 2], c=pred_labels[::skip])

		plt.title('Random drivers - colors are driver ids')
		plt.show()

	for label in range(num_clusters):
		#build a mini-autoencoder here
		indices = np.where(pred_labels == label)[0]
		
		y_train_dense = np.argmax(y_train, axis=1)
		driver_counts = np.zeros(100, dtype='int')
		class_counts = np.zeros(10, dtype='int')
		for ind in indices:
			class_counts[y_train_dense[ind]] += 1
			driver_counts[driver_id_int[ind]] += 1

		print('num indices: %d' % len(indices))
		#print(len(labels_dense))
		print('num train indices: %d' % len(np.where(indices >= 79726)[0]))
		
		unique_drivers = list(map(lambda y: y[0], filter(lambda x: x[1] > 0, enumerate(driver_counts))))
		
		print('driver_counts:')
		print('\n'.join(map(lambda x: str(x[0]) + ' - ' + str(x[1][0]) + ' / ' + str(x[1][1]), filter(lambda y: y[1][0] > 0, enumerate(zip(driver_counts, total_driver_counts))))))
		print('class counts: %s' % str(class_counts))
		print('unique drivers: %s' % str(unique_drivers))

		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')

		if 0:
			ax.scatter(encoded_data[indices,0], encoded_data[indices, 1], zs=encoded_data[indices, 2])#, c=class_labels[indices])
		elif 0:
			label_x_train = np.reshape(to_encode[indices], (len(indices[0]), -1))
			print(label_x_train.shape, to_encode.shape)
			tsne = TSNE(n_components=3)
			tsne_data = tsne.fit_transform(label_x_train)
			ax.scatter(tsne_data[indices,0], tsne_data[indices, 1], zs=tsne_data[indices, 2])#, c=class_labels[indices])

		#plt.legend()
		#plt.title('Clustered driver %d - colors are class labels' % label)
		#plt.show()


def plot_top_clusters(encoder, all_data):
	x_train, y_train, train_id, driver_id, unique_drivers = run_keras_cv_drivers_v2.read_and_normalize_train_data(img_rows, img_cols, color_type)
	to_encode = x_train
	#to_encode = all_data


	# display a 2D plot of the digit classes in the latent space
	encoded_data = encoder.predict(to_encode, batch_size=batch_size)
	num_clusters = 10
	kmeans = KMeans(n_clusters = num_clusters)
	pred_labels = kmeans.fit_predict(encoded_data)

	print(pred_labels[:100])

	if 0:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		n = 5000
		skip = len(encoded_data) / n
		#ax.scatter(encoded_data[::skip, 0], encoded_data[::skip, 1], zs=encoded_data[::skip, 2], c=labels[::skip])
		ax.scatter(encoded_data[::skip, 0], encoded_data[::skip, 1], zs=encoded_data[::skip, 2], c=pred_labels[::skip])

		plt.title('Random drivers - colors are driver ids')
		plt.show()

	for label in range(num_clusters):
		#build a mini-autoencoder here
		indices = np.where(pred_labels == label)[0]
		
		y_train_dense = np.argmax(y_train, axis=1)
		driver_counts = np.zeros(100, dtype='int')
		class_counts = np.zeros(10, dtype='int')
		for ind in indices:
			class_counts[y_train_dense[ind]] += 1
			driver_counts[driver_id_int[ind]] += 1

		print('num indices: %d' % len(indices))
		#print(len(labels_dense))
		print('num train indices: %d' % len(np.where(indices >= 79726)[0]))
		
		unique_drivers = list(map(lambda y: y[0], filter(lambda x: x[1] > 0, enumerate(driver_counts))))
		
		print('driver_counts:')
		print('\n'.join(map(lambda x: str(x[0]) + ' - ' + str(x[1][0]) + ' / ' + str(x[1][1]), filter(lambda y: y[1][0] > 0, enumerate(zip(driver_counts, total_driver_counts))))))
		print('class counts: %s' % str(class_counts))
		print('unique drivers: %s' % str(unique_drivers))

		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')

		if 0:
			ax.scatter(encoded_data[indices,0], encoded_data[indices, 1], zs=encoded_data[indices, 2])#, c=class_labels[indices])
		elif 0:
			label_x_train = np.reshape(to_encode[indices], (len(indices[0]), -1))
			print(label_x_train.shape, to_encode.shape)
			tsne = TSNE(n_components=3)
			tsne_data = tsne.fit_transform(label_x_train)
			ax.scatter(tsne_data[indices,0], tsne_data[indices, 1], zs=tsne_data[indices, 2])#, c=class_labels[indices])

		#plt.legend()
		#plt.title('Clustered driver %d - colors are class labels' % label)
		#plt.show()