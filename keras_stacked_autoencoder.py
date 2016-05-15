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

def get_all_data_autoencoder(  batch_size=10, 
                        epochs = 10, 
                        output_folder = 'keras_stacked_autoencoder',
                        n_hidden = 500,
                        save_filters = True,
                        layer = 0,
                        img_shape = (64, 48, 3),
                        corruption_level = 0.2,
                        all_input_data = None,
                        tie_decoder_weights = True,
                        activation_type = ''
                        ):

	all_input_data = load_all_input_data(img_shape, use_cache = True)


	if n_hidden >= all_input_data.shape[1]:
		print('output size > input size, pca would give perfect reconstruction')
	elif 1: #print out a benchmark pca decomposition
		print('calculating pca mean squared error...')
		pca = PCA(n_components = n_hidden)
		train_components = pca.fit_transform(all_input_data)
		reconstruction = pca.inverse_transform(train_components)
		mse = mean_squared_error(all_input_data, reconstruction)
		print('pca mse (train): %s' % mse)


	input_size = np.prod(all_input_data.shape[1:])

	print('input data shape: %s' % str(all_input_data.shape))

	model = Sequential()
	model.add(Dropout(corruption_level, input_shape=(input_size,)))
	dense_in = Dense(n_hidden)
	model.add(dense_in)

	if tie_decoder_weights:
		model.add(DependentDense(input_size, dense_in))
	else:
		model.add(Dense(input_size))

	model.summary()
	model.compile(RMSprop(), loss='mean_squared_error')

	callbacks = []
	if use_tensorflow: callbacks.append(TensorBoard(log_dir='/tmp/autoencoder'))

	model.fit(all_input_data, all_input_data, 
			nb_epoch=epochs, 
			batch_size=batch_size,
          	callbacks=callbacks
          	)


	

    
    

if __name__ == "__main__":
	autoencoder = get_all_data_autoencoder()
