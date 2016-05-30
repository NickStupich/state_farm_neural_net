from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation
from keras.optimizers import *
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import pylab
import cv2
import os

from sklearn.metrics import mean_squared_error
from run_keras_cv_drivers_v2 import *	


class ConvLayerConfig(object):
	def __init__(self, input_num_filters, output_num_filters, conv_size, stride = 1, activation='relu', dropout_rate = 0.1):
		self.output_num_filters = output_num_filters
		self.input_num_filters = input_num_filters
		self.stride = stride
		self.activation = activation
		self.conv_size = conv_size
		self.dropout_rate = dropout_rate

	def get_encoder_layers(self):
		result = [];
		result.append(Dropout(self.dropout_rate))
		result.append(Convolution2D(self.output_num_filters, 
									self.conv_size, self.conv_size, 
									subsample=(self.stride, self.stride), 
									border_mode='same',
									activation=self.activation))
		return result

	def get_decoder_layers(self):
		result = []
		result.append(UpSampling2D((self.stride, self.stride)))
		result.append(Convolution2D(self.input_num_filters,
									self.conv_size, self.conv_size,
									activation=self.activation, 
									border_mode='same'))
		return result

def build_decoder(layers, input_shape, num_layers = None):
	if num_layers is None:
		num_layers = len(layers)

	input_img = Input(input_shape)
	x = input_img
	for layer in reversed(layers[:num_layers]):
		for sublayer in layer.get_decoder_layers():
			x = sublayer(x)

	model = Model(input_img, x)
	model.summary()

	return model

def build_encoder(layers, input_shape, num_layers = None):
	if num_layers is None:
		num_layers = len(layers)

	input_img = Input(input_shape)
	x = input_img
	for layer in layers[:num_layers]:
		for sublayer in layer.get_encoder_layers():
			x = sublayer(x)

	model = Model(input_img, x)
	model.summary()

	return model

def build_autoencoder(layers, input_shape, num_layers = None):
	if num_layers is None:
		num_layers = len(layers)

	input_img = Input(input_shape)
	x = input_img


	for layer in layers[:num_layers]:
		for sublayer in layer.get_encoder_layers():
			x = sublayer(x)
	
	for layer in reversed(layers[:num_layers]):
		for sublayer in layer.get_decoder_layers():
			x = sublayer(x)

	model = Model(input_img, x)
	model.summary()

	return model

def get_encoder_from_trained_autoencoder(layers, input_shape, autoencoder, num_layers = None):
	encoder = build_encoder(layers, input_shape, num_layers)
	encoder.compile(optimizer='sgd', loss='binary_crossentropy')	#won't be trained

	for encoder_layer, autoencoder_layer in zip(encoder.layers, autoencoder.layers):
		encoder_layer.set_weights(autoencoder_layer.get_weights())

	return encoder

def get_trained_encoder_simple(input_data, data_label, layers, img_rows, img_cols, color_type, nb_epoch=20):
	folder_name = 'trained_simple_%dx%dx%d_%s' % (img_cols, img_rows, color_type, data_label)
	
	if not os.path.exists(folder_name):	os.mkdir(folder_name)

	input_shape = [color_type, img_rows, img_cols]

	autoencoder = build_autoencoder(layers, input_shape)
	autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])


	checkpoint_filename = folder_name + '/weights.{epoch:02d}.hdf5'
	final_weights = folder_name + '/final_weights.hdf5'
	if 0: autoencoder.load_weights(final_weights)
	else:
		callbacks = [
		ModelCheckpoint(checkpoint_filename)
		]
		autoencoder.fit(input_data, input_data,
					batch_size=32,
					nb_epoch=nb_epoch,
					shuffle=True,
					verbose=1,
					callbacks = callbacks
					)
		autoencoder.save_weights(final_weights)

	encoder = get_encoder_from_trained_autoencoder(layers, input_shape, autoencoder)


def main():
	img_rows, img_cols = 128, 96
	color_type = 3
	input_shape = [color_type, img_rows, img_cols]
	conv_size = 3



	layers = []
	layers.append(ConvLayerConfig(color_type, 16, conv_size, 1, 'relu'))
	layers.append(ConvLayerConfig(layers[-1].output_num_filters, 32, conv_size, 2, 'relu'))
	layers.append(ConvLayerConfig(layers[-1].output_num_filters, 64, conv_size, 1, 'relu'))
	layers.append(ConvLayerConfig(layers[-1].output_num_filters, 128, conv_size, 2, 'relu'))

	#encoder = build_encoder(layers, input_shape = (color_type, img_rows, img_cols), num_layers = 2)
	#decoder = build_decoder(layers, input_shape = encoder.output_shape[1:], num_layers = 2)
	#autoencoder = build_autoencoder(layers, input_shape = (color_type, img_rows, img_cols), num_layers = 4)

	if 0:
		train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type)
		input_data = train_data
		data_label='train'
	else:
		input_data = np.zeros((22424+79726, color_type, img_cols, img_rows), dtype='float32')
		input_data[:22424], train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type)
	
		input_data[22424:], test_id = read_and_normalize_test_data(img_rows, img_cols, color_type)
		data_label='all'

	encoder = get_trained_encoder_simple(input_data, data_label, layers, img_rows, img_cols, color_type)


if __name__ == "__main__":
	main()