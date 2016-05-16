if 1:	#imports

	use_tensorflow = False

	import numpy as np
	np.random.seed(2016)

	import os
	import glob
	import cv2
	import math
	import pickle
	import datetime
	import pandas as pd
	import statistics
	import time
	from shutil import copy2
	import warnings
	import random
	warnings.filterwarnings("ignore")

	import sys
	import os
	import timeit

	from utils import tile_raster_images_color

	from sklearn.decomposition import PCA
	from sklearn.metrics import mean_squared_error

	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
	from keras.layers.normalization import BatchNormalization
	from keras.layers.noise import GaussianNoise

	from keras.optimizers import Adam, SGD, RMSprop
	from keras.callbacks import EarlyStopping, ModelCheckpoint
	from keras.utils import np_utils
	from keras.models import model_from_json
	from sklearn.metrics import log_loss
	from scipy.misc import imread, imresize, imshow

	from run_keras_cv_drivers_v2 import *
	from keras_autoencoder_layers import DependentDense, Deconvolution2D, DePool2D

	if use_tensorflow: from keras.callbacks import TensorBoard

"""
THE PLAN:
- get an autoencoder training
- make randomness consistent
- plot filters, make sure reasonable
- convert to class
	- save itself
	- make able to produce a new sequential model using it's own internal (already learned) layers\
- convert to multi-layer
- convert to convnets
- stop decimating input data
- add the random image slice layer
"""
def load_all_input_data(img_shape, flatten = True, use_cache = 1):
	img_cols, img_rows, color_type_global = img_shape

	input_data_cache_fn = 'all_input_data.npy'

	if os.path.exists(input_data_cache_fn) and use_cache:
		all_input_data = np.load(input_data_cache_fn)
	else:
		train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global, one_hot_label_encoding = False)
		test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)


		all_input_data = np.concatenate((train_data, test_data))

		if flatten:
			#TODO: stop reshaping, using convnets
			all_input_data = all_input_data.reshape((all_input_data.shape[0], -1), order='F')

		#TODO: train on all data
		all_input_data = all_input_data[::5]    

		np.save(input_data_cache_fn, all_input_data)

	return all_input_data

class Autoencoder(object):

	def __init__(   self,
                    output_folder = 'keras_stacked_autoencoder',
                    input_shape = 64*48*3,
                    n_hidden = 500,
                    save_filters = True,
                    corruption_level = 0.2,
                    all_input_data = None,
                    tie_decoder_weights = True,
                    activation_type = 'sigmoid'
                    ):

		self.n_hidden = n_hidden
		self.corruption_level = corruption_level

		#self.optimizer = SGD(0.01)
		self.optimizer = RMSprop()
		self.unsupervised_loss = 'mean_squared_error'
		#self.unsupervised_loss = 'binary_crossentropy'

		input_size_flat = np.prod(input_shape)

		self.encoder_layers = []
		model = Sequential()
		model.add(Dropout(corruption_level, input_shape=(input_size_flat,)))
		dense_in = Dense(n_hidden, activation=activation_type)
		model.add(dense_in)

		self.encoder_layers.append(dense_in)

		if tie_decoder_weights:
			decoder = DependentDense(input_size_flat, dense_in, activation=activation_type)
		else:
			decoder = Dense(input_size_flat, activation=activation_type)

		model.add(decoder)

		model.summary()
		model.compile(self.optimizer, loss=self.unsupervised_loss, metrics=['mean_squared_error', 'binary_crossentropy'])

		callbacks = []
		if use_tensorflow: callbacks.append(TensorBoard(log_dir='/tmp/autoencoder'))

		self.model = model
		self.callbacks = callbacks

		self.encoder_model = None

	def get_weights_filename(self):
		fn = 'autoencoder_weights_hidden%s_corruption%s.weights' % (str(self.n_hidden), str(self.corruption_level))
		return fn

	def load_weights_from_file(self):
		fn = self.get_weights_filename()
		if os.path.exists(fn):
			self.model.load_weights(fn)
			self.model.compile(self.optimizer, loss=self.unsupervised_loss)
			return True
		else:
			return False

	def train(self, 
				input_data, 
				print_pca_mse = True, 
                epochs = 10,
				batch_size=10, 
				):

		if self.n_hidden >= input_data.shape[1]:
			print('output size > input size, pca would give perfect reconstruction')
		elif print_pca_mse: #print out a benchmark pca decomposition
			print('calculating pca mean squared error...')
			pca = PCA(n_components = self.n_hidden)
			train_components = pca.fit_transform(input_data)
			reconstruction = pca.inverse_transform(train_components)
			mse = mean_squared_error(input_data, reconstruction)
			print('pca mse (train): %s' % mse)


		print('input data shape: %s' % str(input_data.shape))

		self.model.fit(input_data, input_data, 
				nb_epoch=epochs, 
				batch_size=batch_size,
	          	callbacks=self.callbacks
	          	)

		fn = self.get_weights_filename()
		self.model.save_weights(fn, overwrite=True)

	def get_encoder_model(self, print_summary = True, force_new=False):
		if self.encoder_model is None or force_new:
			self.encoder_model = Sequential()
			for layer in self.encoder_layers:
				self.encoder_model.add(layer)

			for i, layer in enumerate(self.encoder_layers):
				weights = self.encoder_layers[i].get_weights()
				if len(weights):
					self.encoder_model.layers[i].set_weights(weights)

			if print_summary: self.encoder_model.summary()
			self.encoder_model.compile(self.optimizer, loss=self.unsupervised_loss) #shouldnt do any training on here

		return self.encoder_model

	def encode(self, input_data):
		model = self.get_encoder_model()
		result = model.predict(input_data)
		return result

def get_full_model(autoencoder, freeze_lower_layers=True):
	encoder_model = autoencoder.get_encoder_model(force_new=True)

	full_model = encoder_model
	
	if freeze_lower_layers:	#lock weights from pre-trained layers while bottom layer(s) train
		for layer in full_model.layers:
			layer.trainable_weights = []

	if 0:
		full_model.add(Dropout(0.5))
		full_model.add(Dense(50, activation='sigmoid'))
		full_model.add(Dropout(0.5))

	full_model.add(Dense(10))

	full_model.add(Activation('softmax'))

	full_model.summary()
	
	full_model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy')
	#full_model.compile(optimizer='sgd', loss='categorical_crossentropy')

	return full_model
	

if __name__ == "__main__":
	
	img_shape = (64, 48, 3)

	force_retrain_model = False
	autoencoder = Autoencoder()	
	if autoencoder.load_weights_from_file() and not force_retrain_model:
		print('loaded autoencoder weights from file')
	else:
		print('Calculating autoencoder weights')
		all_input_data = load_all_input_data(img_shape, 
											flatten=True, 
											use_cache = True)
		input_size = np.prod(all_input_data.shape[1:])
		autoencoder.train(all_input_data, epochs=200, print_pca_mse=True)




	
	img_cols, img_rows, color_type_global = img_shape
	nfolds = 13
	layer = 0
	nb_epoch = 100
	batch_size = 64
	random_state = 51

	img_cols, img_rows, color_type_global = img_shape

	train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global, one_hot_label_encoding = True)
	train_data = train_data.reshape((train_data.shape[0], -1), order='F')

	kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
	num_fold = 0
	sum_score = 0
	for train_drivers, test_drivers in kf:

		unique_list_train = [unique_drivers[i] for i in train_drivers]
		X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
		unique_list_valid = [unique_drivers[i] for i in test_drivers]
		X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

		num_fold += 1
		print('Start KFold number {} from {}'.format(num_fold, nfolds))
		print('Split train: ', len(X_train), len(Y_train))
		print('Split valid: ', len(X_valid), len(Y_valid))
		print('Train drivers: ', unique_list_train)
		print('Test drivers: ', unique_list_valid)

		model = get_full_model(autoencoder, freeze_lower_layers=True)

		kfold_weights_path = os.path.join('cache', 'weights_kfold_' + str(num_fold) + '.h5')
		#if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:

		callbacks = [
		    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
		    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
		]
		model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
		      shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
		      callbacks=callbacks)

		if os.path.isfile(kfold_weights_path):
		    model.load_weights(kfold_weights_path)

		predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
		score = log_loss(Y_valid, predictions_valid)
		print('Score log_loss: ', score)
		sum_score += score*len(test_index)

		# Store valid predictions
		# for i in range(len(test_index)):
		#     yfull_train[test_index[i]] = predictions_valid[i]

		# # Store test predictions
		# test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
		# yfull_test.append(test_prediction)

	score = sum_score/len(train_data)
	print("Log_loss train independent avg: ", score)
