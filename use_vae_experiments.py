import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import objectives
from keras.layers import Flatten, Dropout, Activation, Reshape, Merge
from keras.optimizers import *

import run_keras_cv_drivers_v2
import train_data_generator

from sklearn.cluster import KMeans

predict_batch_size = 256

def cluster_and_classify(encoder, all_data, img_rows, img_cols, color_type):
	x_train, y_train, train_id, driver_id, unique_drivers = run_keras_cv_drivers_v2.read_and_normalize_train_data(img_rows, img_cols, color_type)
	driver_id_int = list(map(lambda x: int(x.strip('p')), driver_id))
	total_driver_counts = np.zeros(100, dtype='int')
	for i in driver_id_int: total_driver_counts[i] += 1
	to_encode = x_train
	#to_encode = all_data


	# display a 2D plot of the digit classes in the latent space
	encoded_data = encoder.predict(to_encode, batch_size=predict_batch_size, verbose=True)
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

def cluster_and_classify2(encoder, encode_layers, img_rows, img_cols, color_type, latent_dim=3):
	n_folds=2
	n_clusters=4
	batch_size=32
	nb_epoch = 50

	data_iterator = train_data_generator.driver_split_data_generator(n_folds, img_rows, img_cols, color_type)

	for fold, data_provider in enumerate(data_iterator):
		(X_train, Y_train, X_valid, Y_valid) = data_provider()

		mean_pixel = [103.939, 116.779, 123.68] #vgg16 paper
		for c in range(3):
			X_train[:, c, :, :] = (X_train[:, c, :, :] + mean_pixel[c])/255.
			X_valid[:, c, :, :] = (X_valid[:, c, :, :] + mean_pixel[c])/255.

		print(X_train.dtype)

		X_train_encoded = encoder.predict(X_train, batch_size=predict_batch_size, verbose=True)
		X_valid_encoded = encoder.predict(X_valid, batch_size=predict_batch_size, verbose=True)

		kmeans = KMeans(n_clusters = n_clusters)
		train_data_clusters = kmeans.fit_predict(X_train_encoded)
		valid_data_clusters = kmeans.predict(X_valid_encoded)

		for cluster in range(n_clusters):
			train_indices = np.where(train_data_clusters == cluster)[0]
			valid_indices = np.where(valid_data_clusters == cluster)[0]

			X_train_cluster = X_train[train_indices]
			Y_train_cluster = Y_train[train_indices]
			X_valid_cluster = X_valid[valid_indices]
			Y_valid_cluster = Y_valid[valid_indices]


			x_input = Input(shape=(color_type, img_rows, img_cols))
			x = x_input
			for encode_layer in encode_layers:
			    x = encode_layer(x)

			x = Dense(latent_dim)(x)

			output = Dense(10)(x)

			model = Model(x_input, output)
			#model.summary()

			optimizer = SGD(lr=1e-4)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy')

			for new_layer, encoder_layer in zip(model.layers[1:], encoder.layers[1:]):
				if new_layer.get_weights() and encoder_layer.get_weights() and new_layer.get_weights()[0].shape == encoder_layer.get_weights()[0].shape:
					new_layer.set_weights(encoder_layer.get_weights())

			callbacks = []
			model.fit(X_train_cluster, Y_train_cluster,
					batch_size=batch_size,
					verbose=1,
					nb_epoch = nb_epoch,
					shuffle=True,
					validation_data = (X_valid_cluster, Y_valid_cluster),
					callbacks = callbacks)


def plot_top_clusters(encoder, all_data, img_rows, img_cols, color_type):
	x_train, y_train, train_id, driver_id, unique_drivers = run_keras_cv_drivers_v2.read_and_normalize_train_data(img_rows, img_cols, color_type)

	driver_labels = np.argmax(y_train, axis=1)
	class_labels = np.array(list(map(lambda s: int(s.strip('p')), driver_id)))

	# display a 2D plot of the digit classes in the latent space
	encoded_train_data = encoder.predict(all_data[79726:], batch_size=predict_batch_size)
	num_clusters = 10
	kmeans = KMeans(n_clusters = num_clusters)
	train_data_clusters = kmeans.fit_predict(encoded_train_data)

	encoded_test_data = encoder.predict(all_data[:79726], batch_size=predict_batch_size, verbose=True)
	test_data_clusters = kmeans.predict(encoded_test_data)

	if 1:
		fig = plt.figure()
		n = 2000
		skip = len(encoded_train_data) / n

		ax1 = fig.add_subplot(121, projection='3d')
		train_colors = driver_labels
		train_colors = train_data_clusters
		ax1.scatter(encoded_train_data[::skip, 0], encoded_train_data[::skip, 1], zs=encoded_train_data[::skip, 2], c=train_colors[::skip])

		skip = len(encoded_test_data) / n
		ax2 = fig.add_subplot(122, projection='3d')
		test_colors = test_data_clusters
		ax2.scatter(encoded_test_data[::skip, 0], encoded_test_data[::skip, 1], zs=encoded_test_data[::skip, 2], c=test_colors[::skip])

		plt.title('Random drivers - colors are driver ids')
		plt.show()

	for label in range(num_clusters):


		fig = plt.figure()
		#build a mini-autoencoder here
		indices = np.where(train_data_clusters == label)[0]

		test_indices = np.where(test_data_clusters == label)[0]

		ax1 = fig.add_subplot(211, projection='3d')
		ax1.scatter(encoded_train_data[indices,0], encoded_train_data[indices, 1], zs=encoded_train_data[indices, 2], c=class_labels[indices])

		ax2 = fig.add_subplot(212, projection='3d')
		ax2.scatter(encoded_test_data[test_indices,0], encoded_test_data[test_indices, 1], zs=encoded_test_data[test_indices, 2])

		plt.legend()
		plt.title('Clustered driver %d - colors are class labels' % label)
		plt.show()
