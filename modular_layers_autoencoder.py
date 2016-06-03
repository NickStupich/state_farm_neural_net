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
from pretrained_vgg16 import vgg_std16_model

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
									activation=self.activation,
									init='he_normal'
									))
		return result

	def get_decoder_layers(self):
		result = []
		result.append(UpSampling2D((self.stride, self.stride)))
		result.append(Convolution2D(self.input_num_filters,
									self.conv_size, self.conv_size,
									activation=self.activation, 
									border_mode='same',
									init='he_normal'
									))
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
	#model.summary()

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
	#model.summary()

	return model

def build_autoencoder(layers, input_shape, num_layers = None):
	if num_layers is None:
		num_layers = len(layers)

	input_img = Input(input_shape)
	x = input_img


	for layer in layers[:num_layers]:
		for sublayer in layer.get_encoder_layers():
			x = sublayer(x)

	x = Dropout(0.5)(x)
	
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

def get_trained_encoder_simple(input_data, layers, img_rows, img_cols, color_type, nb_epoch=20, folder_name='folder_name'):

	input_shape = [color_type, img_rows, img_cols]

	autoencoder = build_autoencoder(layers, input_shape)
	autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse'])


	checkpoint_filename = folder_name + '/weights.{epoch:02d}.hdf5'
	final_weights = folder_name + '/final_weights.hdf5'
	if os.path.exists(final_weights) and 1: 
		autoencoder.load_weights(final_weights)
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

	return encoder

printedSummary = False
def create_logistic_model(encoded_shape):
    global printedSummary
    model = Sequential()
    model.add(Flatten(input_shape = (encoded_shape)))
    #model.add(Dropout(0.5))

    # model.add(Dense(100))
    # model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    if not printedSummary:
        model.summary()
        printedSummary = True

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
    return model

def create_conv_model(encoded_shape):
	global printedSummary
	model = Sequential()
	model.add(Dropout(0.5, input_shape = (encoded_shape)))

	model.add(Convolution2D(512, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(0.5))
	
	model.add(Convolution2D(512, 3, 3, border_mode='same', init='he_normal', subsample=(2,2)))
	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(100))
	model.add(Dropout(0.5))

	model.add(Dense(10))
	model.add(Activation('softmax'))

	if not printedSummary:
	    model.summary()
	    printedSummary = True

	model.compile(Adam(lr=5e-4), loss='categorical_crossentropy')
	return model

def create_run_keras_skip_first(encoded_shape):
	global printedSummary
	model = Sequential()
	model.add(Dropout(0.5, input_shape = (encoded_shape)))

	model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
	model.add(MaxPooling2D(pool_size=(8, 8)))
	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(10))
	model.add(Activation('softmax'))

	if not printedSummary:
	    model.summary()
	    printedSummary = True

	model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
	return model


def cross_validation_wth_encoder_no_finetune(encoder, 
											img_shape, 
											nfolds=13, 
											do_test_predictions = False, 
											folder_name='folder_name', 
											model_build_func = create_logistic_model,
											retrain_single_model = True):
	nb_epoch = 100
	batch_size = 128
	random_state = 51
	restore_from_last_checkpoint = 0

	train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(*img_shape)
	if do_test_predictions:
		test_data, test_id = read_and_normalize_test_data(*img_shape)
		encoded_test_data = encoder.predict(test_data)

	encoded_train_data = encoder.predict(train_data)
	encoded_shape = encoded_train_data.shape[1:]
	print('encoded shape: %s' % str(encoded_shape))
	train_data = None


	weights_folder = '%s/cache%s' % (folder_name, 'encoder_no_finetune')
	if not os.path.exists(weights_folder): os.mkdir(weights_folder)

	if retrain_single_model:
		model = model_build_func(encoded_shape)

	yfull_train = dict()
	yfull_test = []
	kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
	num_fold = 0
	sum_score = 0
	for train_drivers, test_drivers in kf:

		if not retrain_single_model:
			model = model_build_func(encoded_shape)

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

		kfold_weights_path = os.path.join(weights_folder, 'weights_kfold_' + str(num_fold) + '.h5')
		if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
			callbacks = [
				EarlyStopping(monitor='val_loss', patience=5, verbose=0),
				ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
			]
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
				shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
				callbacks=callbacks)
		if os.path.isfile(kfold_weights_path):
		    model.load_weights(kfold_weights_path)

		# score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
		# print('Score log_loss: ', score[0])

		predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
		score = log_loss(Y_valid, predictions_valid)
		print('Score log_loss: ', score)
		sum_score += score*len(test_index)

		# Store valid predictions
		for i in range(len(test_index)):
			yfull_train[test_index[i]] = predictions_valid[i]

		# Store test predictions
		if do_test_predictions:
			test_prediction = model.predict(encoded_test_data, batch_size=batch_size, verbose=1)
			yfull_test.append(test_prediction)

	score = sum_score/len(encoded_train_data)
	print("Log_loss train independent avg: ", score)

	predictions_valid = get_validation_predictions(encoded_train_data, yfull_train)
	score1 = log_loss(train_target, predictions_valid)
	if abs(score1 - score) > 0.0001:
	    print('Check error: {} != {}'.format(score, score1))

	print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
	info_string = 'loss_' + str(score) \
			+ '_r_' + str(img_rows) \
			+ '_c_' + str(img_cols) \
			+ '_folds_' + str(nfolds) \
			+ '_ep_' + str(nb_epoch)

	test_res = merge_several_folds_mean(yfull_test, nfolds)
	# test_res = merge_several_folds_geom(yfull_test, nfolds)

	if do_test_predictions:
		create_submission(test_res, test_id, info_string)

	save_useful_data(predictions_valid, train_id, model, info_string)

def plot_encoder_things(encoder, data, output, subsample=100):
	data = data[::subsample]
	output = output[::subsample]

	encoded_data = encoder.predict(data)

	
	if 0: #plot some feature images
		for img, encoding, label in zip(data, encoded_data, output):
			if img.shape[0] == 1:
				img_reshape = np.reshape(img[0], (img.shape[2], img.shape[1]))
			else:
				img_reshape = np.transpose(img, (2, 1, 0))
			print(img_reshape.shape)

			num_to_show = 4

			pylab.subplot(num_to_show, num_to_show, 1)
			pylab.imshow(img_reshape, cmap='gray')

			for i in range(num_to_show):
				for j in range(num_to_show):
					if i == 0 and j == 0:
						continue
					n = i*num_to_show + j

					e = encoding[n]
					e -= np.mean(e)
					e *= np.std(e)
					e += 0.5
					e = np.clip(e, 0.0, 1.0)

					e = np.reshape(e, (e.shape[1], e.shape[0]))

					pylab.subplot(num_to_show, num_to_show, n+1)

					pylab.imshow(e, cmap='gray')

			pylab.show()

	if 1: #plot conv filters
		conv_layer = encoder.layers[2]
		weights = conv_layer.get_weights()[0] #[1] is the biases

		n = 4

		for i in range(n):
			for j in range(n):
				index = i*n+j
				if weights.shape[1] == 3:
					w_reshape = np.transpose(weights[index], (2, 1, 0))
				else:
					w_reshape = weights[index][0]

				pylab.subplot(n, n, index+1)
				pylab.imshow(w_reshape, cmap='gray')

		pylab.show()

	#pylab.scatter(encoded_data[:, i], encoded)

def main():
	img_rows, img_cols = 128, 96
	# img_rows, img_cols = 64, 64
	# img_rows, img_cols = 64, 48
	color_type = 1
	input_shape = [img_rows, img_cols, color_type]
	conv_size = 3

	nb_epoch = 20

	need_to_train = True #if false and you do, stuff will fail

	layers = []
	if 0:
		layers.append(ConvLayerConfig(color_type, 16, conv_size, 1, 'relu'))
		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 32, conv_size, 2, 'relu'))
		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 64, conv_size, 1, 'relu'))
		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 128, conv_size, 2, 'relu'))
		model_name='test1'
		supervised_model_builder = create_logistic_model
	elif 0:
		layers.append(ConvLayerConfig(color_type, 32, conv_size, 1, 'relu'))
		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 32, conv_size, 2, 'relu'))

		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 64, conv_size, 1, 'relu'))
		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 64, conv_size, 2, 'relu'))

		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 128, conv_size, 1, 'relu'))
		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 128, conv_size, 2, 'relu'))
	
		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 256, conv_size, 1, 'relu'))
		layers.append(ConvLayerConfig(layers[-1].output_num_filters, 256, conv_size, 2, 'relu'))

		model_name = 'vgg16_small'
		supervised_model_builder = create_logistic_model
	elif 0:
		layers.append(ConvLayerConfig(color_type, 32, conv_size, 2, 'relu'))
		model_name = 'single_layer_run_keras_script'
		supervised_model_builder = create_run_keras_skip_first
		nb_epoch = 5
	elif 0:
		layers.append(ConvLayerConfig(color_type, 16, 11, 2, 'relu'))
		model_name = 'single_large_layer'
		supervised_model_builder = create_run_keras_skip_first
		nb_epoch = 5

	elif 1:
		layers.append(ConvLayerConfig(color_type, 16, 11, 2, 'relu', dropout_rate = 0.2))
		model_name = 'single_large_layer_dropout0.2'
		supervised_model_builder = create_run_keras_skip_first
		nb_epoch = 50
	
	#encoder = build_encoder(layers, input_shape = (color_type, img_rows, img_cols), num_layers = 2)
	#decoder = build_decoder(layers, input_shape = encoder.output_shape[1:], num_layers = 2)
	#autoencoder = build_autoencoder(layers, input_shape = (color_type, img_rows, img_cols), num_layers = 4)


	if 1:
		if need_to_train:
			train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(*input_shape)
		else:
			input_data = None

		input_data = train_data
		data_label='train'
	else:
		data_label= 'all'
		if need_to_train:
			input_data = np.zeros((22424+79726, *(np.array(input_shape)[[2, 1, 0]])), dtype='float32')
			input_data[:22424], train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(*input_shape)
			input_data[22424:], test_id = read_and_normalize_test_data(*input_shape)
		else:
			input_data = None
	
	folder_name = 'trained_simple_%dx%dx%d_%s_%s' % (img_cols, img_rows, color_type, model_name, data_label)	
	if not os.path.exists(folder_name):	os.mkdir(folder_name)

	encoder = get_trained_encoder_simple(input_data, 
									layers, 
									img_rows, img_cols, color_type, 
									nb_epoch=nb_epoch, 
									folder_name=folder_name)

	if 1:
		train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(*input_shape)
		plot_encoder_things(encoder, train_data, train_target)
	else:
		#run gc on input data, not being used anymore
		input_data = None
		cross_validation_wth_encoder_no_finetune(encoder, input_shape, 
										nfolds=13, 
										do_test_predictions = True,
										folder_name=folder_name, 
										model_build_func = supervised_model_builder,
										retrain_single_model = True)

	return encoder

if __name__ == "__main__":
	main()
