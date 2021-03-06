from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation
from keras.optimizers import *
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import pylab
import cv2

from sklearn.metrics import mean_squared_error

from run_keras_cv_drivers_v2 import *	

kernel_size = 5
pool_size = 4

def get_simple_autoencoder(data = None, img_rows = 64, img_cols = 64, color_type = 3, hidden_depth = 32, layer = 0):
	input_img = Input((color_type, img_rows, img_cols))
	x = input_img
	conv_size = 3
	
	x = Dropout(0.1)(x)

	x = Convolution2D(hidden_depth, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((pool_size, pool_size))(x)

	x = UpSampling2D((pool_size, pool_size))(x)
	x = Convolution2D(color_type, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)

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
					nb_epoch=50,
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
	# all_input_data = np.concatenate([train_data, test_data])

	model = get_simple_autoencoder(all_input_data, img_rows, img_cols, color_type)

	input_img = train_data[0]

	input_img = Input((color_type, img_rows, img_cols))

	x = input_img
	x = Convolution2D(32, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(input_img)
	x = MaxPooling2D((pool_size, pool_size))(x)
	encoder = Model(input_img, x)
	encoder.summary()
	encoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])
	encoder.layers[1].set_weights(model.layers[2].get_weights())

	decoder_input = Input((32, img_rows/2, img_cols/2))
	x = decoder_input
	x = UpSampling2D((pool_size, pool_size))(x)
	x = Convolution2D(color_type, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)

	decoder = Model(decoder_input, x)
	decoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])
	decoder.layers[-1].set_weights(model.layers[-1].get_weights())



	if 1:
		print('predicting reconstruction and encoding...')
		encoded_train_data = encoder.predict(train_data)
		reconstructed_train_data = decoder.predict(encoded_train_data)
		print('done')
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

	x = Convolution2D(32, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((pool_size, pool_size))(x)

	x = Dropout(0.1)(x)	#needed? at least to maintain consistency with previous second autoencoder

	x = Convolution2D(64, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((pool_size, pool_size))(x)

	#"bottleneck"

	x = UpSampling2D((pool_size, pool_size))(x)
	x = Convolution2D(32, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)

	x = UpSampling2D((pool_size, pool_size))(x)
	x = Convolution2D(color_type, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)


	two_layer_model = Model(input_img, x)
	two_layer_model.summary()


	two_layer_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])
	
	if 0:
		print('score before setting weights: ')
		print(two_layer_model.evaluate(train_data, train_data, batch_size = 16))

	two_layer_model.layers[2].set_weights(model.layers[2].get_weights())
	two_layer_model.layers[5].set_weights(model2.layers[2].get_weights())
	two_layer_model.layers[8].set_weights(model2.layers[5].get_weights())
	two_layer_model.layers[10].set_weights(model.layers[5].get_weights())

	if 0:
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
					nb_epoch=100,
					shuffle=True,
					verbose=1,
					callbacks = callbacks)


	if 0:
		print('score after tuning weights: ')
		print(two_layer_model.evaluate(train_data, train_data, batch_size=16))


	#get a sample image from teh 2 layer net
	if 0:
		print('getting two layer model reconstruction...')
		two_layer_reconstruction = two_layer_model.predict(train_data)
		print('done')

		decimate_factor = int((pool_size**2) * 2)
		cv_images = map(lambda img: np.transpose(img, (1, 2, 0)), all_input_data)
		small_cv_images = map(lambda img: cv2.resize(img, (img_rows//decimate_factor, img_cols//decimate_factor)), cv_images)
		big_cv_images = map(lambda img: cv2.resize(img, (img_rows, img_cols)), small_cv_images)
		flattened_images = map(np.ndarray.flatten, big_cv_images)
		mse = mean_squared_error(np.reshape(np.transpose(all_input_data, (0, 2, 3, 1)), (all_input_data.shape[0], -1)), np.array(list(flattened_images)))
		print('mean squared interpolation error: %s' % str(mse))

	if 0:

		for i in range(100):
			in_img0 = train_data[i]
			out_2layer_img0 = two_layer_reconstruction[i]

			pylab.subplot(211)
			pylab.imshow(np.transpose(in_img0, (1, 2, 0)))
			pylab.subplot(212)
			pylab.imshow(np.transpose(out_2layer_img0, (1, 2, 0)))
			pylab.show()



	if 1:

		#try to make some predictions off of the encoder
		input_img = Input((color_type, img_rows, img_cols))
		x = input_img

		x = Dropout(0.1)(x)

		x = Convolution2D(32, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)
		x = MaxPooling2D((pool_size, pool_size))(x)

		x = Dropout(0.1)(x)	#needed? at least to maintain consistency with previous second autoencoder

		x = Convolution2D(64, kernel_size, kernel_size, activation='relu', border_mode='same', init='he_normal')(x)
		x = MaxPooling2D((pool_size, pool_size))(x)


		two_layer_encoder = Model(input_img, x)
		two_layer_encoder.summary()


		two_layer_encoder.compile(optimizer='rmsprop', loss='binary_crossentropy') #doesn't matter, we're not learning

		two_layer_encoder.layers[2].set_weights(two_layer_model.layers[2].get_weights())
		two_layer_encoder.layers[5].set_weights(two_layer_model.layers[5].get_weights())

		encoded_train_data = two_layer_encoder.predict(train_data)#use just train data, we're not learning

		print('encoded train data shape: %s' % str(encoded_train_data.shape))

		def create_logistic_model(input_shape):
			model = Sequential()
			model.add(Flatten(input_shape = input_shape))
			model.add(Dropout(0.5))

			model.add(Dense(50, activation='sigmoid'))
			model.add(Dropout(0.5))

			model.add(Dense(10))
			model.add(Activation('softmax'))

			model.summary()

			model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
			return model




		nfolds = 13
		layer = 0
		nb_epoch = 100
		batch_size = 64
		random_state = 51

		_, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type, one_hot_label_encoding = True)

		kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
		num_fold = 0
		sum_score = 0
		for train_drivers, test_drivers in kf:

			unique_list_train = [unique_drivers[i] for i in train_drivers]
			X_train, Y_train, train_index = copy_selected_drivers(encoded_train_data, train_target, driver_id, unique_list_train)
			unique_list_valid = [unique_drivers[i] for i in test_drivers]
			X_valid, Y_valid, test_index = copy_selected_drivers(encoded_train_data, train_target, driver_id, unique_list_valid)

			num_fold += 1
			print('Start KFold number {} from {}'.format(num_fold, nfolds))
			print('Split train: ', len(X_train), len(Y_train))
			print('Split valid: ', len(X_valid), len(Y_valid))
			print('Train drivers: ', unique_list_train)
			print('Test drivers: ', unique_list_valid)


			model = create_logistic_model(encoded_train_data.shape[1:])


			kfold_weights_path = os.path.join('cache', 'weights_kfold_' + str(num_fold) + '.h5')

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

		score = sum_score/len(train_data)
		print("Log_loss train independent avg: ", score)
