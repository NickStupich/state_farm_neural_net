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
		all_input_data = all_input_data[::50]    

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
		model.compile(RMSprop(), loss='mean_squared_error')

		callbacks = []
		if use_tensorflow: callbacks.append(TensorBoard(log_dir='/tmp/autoencoder'))

		self.model = model
		self.callbacks = callbacks

		self.encoder_model = None

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

	def get_encoder_model(self):
		if self.encoder_model is None:
			self.encoder_model = Sequential()
			for layer in self.encoder_layers:
				self.encoder_model.add(layer)

			self.encoder_model.summary()
			self.encoder_model.compile(RMSprop(), loss='mean_squared_error') #shouldnt do any training on here

			for i, layer in enumerate(self.encoder_layers):
				weights = self.encoder_layers[i].get_weights()
				if len(weights):
					self.encoder_model.layers[i].set_weights(weights)

		return self.encoder_model

	def encode(self, input_data):
		model = self.get_encoder_model()
		result = model.predict(input_data)
		return result

if __name__ == "__main__":
	
	img_shape = (64, 48, 3)

	all_input_data = load_all_input_data(img_shape, 
										flatten=True, 
										use_cache = True)

	input_size = np.prod(all_input_data.shape[1:])

	autoencoder = Autoencoder()
	autoencoder.train(all_input_data, epochs=20, print_pca_mse=False)
	encoded_input_data = autoencoder.encode(all_input_data)
	print('encoded data shape: %s' % str(encoded_input_data.shape))

	encoder_model = autoencoder.get_encoder_model()

	full_model = encoder_model
	full_model.add(Dense(10))

	full_model.add(Activation('softmax'))


	print('compiling and summarizing full model')
	full_model.compile(RMSprop(), loss='categorical_crossentropy')
	full_model.summary()