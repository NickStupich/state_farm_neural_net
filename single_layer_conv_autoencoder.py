from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation
from keras.optimizers import *
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import pylab

from sklearn.metrics import mean_squared_error

from run_keras_cv_drivers_v2 import *	

def get_simple_autoencoder(data = None, img_rows = 64, img_cols = 64, color_type = 3, hidden_depth = 32, layer = 0):
	input_img = Input((color_type, img_rows, img_cols))
	x = input_img
	conv_size = 3
	
	x = Dropout(0.1)(x)

	x = Convolution2D(hidden_depth, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((2, 2))(x)

	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(color_type, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)

	model = Model(input_img, x)

	model.summary()
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])

	model_weights_path = 'single_layer_conv_weights_layer%d.h5' % layer
	if os.path.isfile(model_weights_path):
		model.load_weights(model_weights_path)
	else:
		callbacks = [
			    ModelCheckpoint(model_weights_path, monitor='val_loss', save_best_only=False, verbose=0),
		]
		model.fit(data, data,
					batch_size = 128,
					nb_epoch=10,
					shuffle=True,
					verbose=1,
					callbacks = callbacks)

	return model

if __name__ == "__main__":

	img_rows, img_cols = 128, 96
	color_type = 3


	train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type)
	#test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type)
	all_input_data = train_data
	#all_input_data = np.concatenate([train_data, test_data])

	model = get_simple_autoencoder(all_input_data, img_rows, img_cols, color_type)

	input_img = train_data[0]

	input_img = Input((color_type, img_rows, img_cols))

	x = input_img
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(input_img)
	x = MaxPooling2D((2, 2))(x)
	encoder = Model(input_img, x)
	encoder.summary()
	encoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])
	encoder.layers[1].set_weights(model.layers[2].get_weights())

	decoder_input = Input((32, img_rows/2, img_cols/2))
	x = decoder_input
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(color_type, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)

	decoder = Model(decoder_input, x)
	decoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])
	decoder.layers[-1].set_weights(model.layers[-1].get_weights())


	print('predicting reconstruction and encoding...')
	encoded_train_data = encoder.predict(train_data)
	reconstructed_train_data = decoder.predict(encoded_train_data)
	print('done')

	if 0:
		in_img0 = train_data[0]
		out_img0 = reconstructed_train_data[0]

		pylab.subplot(211)
		pylab.imshow(np.transpose(in_img0, (1, 2, 0)))
		pylab.subplot(212)
		pylab.imshow(np.transpose(out_img0, (1, 2, 0)))
		pylab.show()

	model2 = get_simple_autoencoder(encoded_train_data, encoded_train_data.shape[3], encoded_train_data.shape[2], encoded_train_data.shape[1], hidden_depth=64, layer = 1)

	if 0:
		print('getting decoded model2 predictions, and decoding them using model 1 decoder to get back to original...')
		double_encoded_once_decoded_train_data = model2.predict(encoded_train_data)
		reconstructed2_train_data = decoder.predict(double_encoded_once_decoded_train_data)
		print('done')

		try:
			overall_mse = mean_squared_error(np.ndarray.flatten(reconstructed2_train_data), np.ndarray.flatten(train_data))
			print('overall 2 layer mse: %s' % str(overall_mse))
		except Exception as e:
			print(e)

		if 0:
			in_img0 = train_data[0]
			out2_img0 = reconstructed2_train_data[0]

			pylab.subplot(211)
			pylab.imshow(np.transpose(in_img0, (1, 2, 0)))
			pylab.subplot(212)
			pylab.imshow(np.transpose(out2_img0, (1, 2, 0)))
			pylab.show()



	#build up a 2 conv layer model, using previously learned weights
	input_img = Input((color_type, img_rows, img_cols))
	x = input_img

	x = Dropout(0.1)(x)

	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((2, 2))(x)

	x = Dropout(0.1)(x)	#needed? at least to maintain consistency with previous second autoencoder

	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((2, 2))(x)

	#"bottleneck"

	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)

	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(color_type, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)


	two_layer_model = Model(input_img, x)
	two_layer_model.summary()


	two_layer_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])
	print('score before setting weights: ')
	print(two_layer_model.evaluate(train_data, train_data, batch_size = 16))

	two_layer_model.layers[2].set_weights(model.layers[2].get_weights())
	two_layer_model.layers[5].set_weights(model2.layers[2].get_weights())
	two_layer_model.layers[8].set_weights(model2.layers[5].get_weights())
	two_layer_model.layers[10].set_weights(model.layers[5].get_weights())

	print('score with greedy weights: ')
	print(two_layer_model.evaluate(train_data, train_data, batch_size=16))

	model_weights_path = 'two_layer_conv_weights_layer.h5'
	if os.path.isfile(model_weights_path):
		two_layer_model.load_weights(model_weights_path)
	else:
		callbacks = [
			    ModelCheckpoint(model_weights_path, monitor='val_loss', save_best_only=False, verbose=0),
		]
		two_layer_model.fit(train_data, train_data,
					batch_size = 64,
					nb_epoch=10,
					shuffle=True,
					verbose=1,
					callbacks = callbacks)



	print('score after tuning weights: ')
	print(two_layer_model.evaluate(train_data, train_data, batch_size=16))


	#get a sample image from teh 2 layer net

	print('getting two layer model reconstruction...')
	two_layer_reconstruction = two_layer_model.predict(train_data)
	print('done')

	if 1:
		in_img0 = train_data[0]
		out_2layer_img0 = two_layer_reconstruction[0]

		pylab.subplot(211)
		pylab.imshow(np.transpose(in_img0, (1, 2, 0)))
		pylab.subplot(212)
		pylab.imshow(np.transpose(out_2layer_img0, (1, 2, 0)))
		pylab.show()