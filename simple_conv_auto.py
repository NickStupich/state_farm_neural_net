from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

def get_conv_autoencoder_model(input_img, corruption_level = 0.5, color_type=3, activation='linear'):	

	x = input_img

	#x = Dropout(corruption_level)(x)
	x = Convolution2D(32, 3, 3, activation=activation, border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((2, 2))(x)
	


	x = Dropout(corruption_level)(x)
	x = Convolution2D(64, 3, 3, activation=activation, border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((2, 2))(x)
	
	x = Dropout(corruption_level)(x)
	x = Convolution2D(128, 3, 3, activation=activation, border_mode='same', init='he_normal')(x)
	x = MaxPooling2D((8, 8))(x)

	# x = Dropout(corruption_level)(x)
	# x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
	# x = MaxPooling2D((4, 4), border_mode='same')(x)

	# #here's the middle between encoder & decoder
	encoded = x

	# x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
	# x = UpSampling2D((4, 4))(x)	

	x = Convolution2D(128, 3, 3, activation=activation, border_mode='same', init='he_normal')(x)
	x = UpSampling2D((8, 8))(x)
	
	x = Convolution2D(64, 3, 3, activation=activation, border_mode='same', init='he_normal')(x)
	x = UpSampling2D((2, 2))(x)
	
	x = Convolution2D(32, 3, 3, activation=activation, border_mode='same', init='he_normal')(x)
	x = UpSampling2D((2, 2))(x)
	
	decoded = Convolution2D(color_type, 3, 3, activation='sigmoid', border_mode='same')(x)

	autoencoder = Model(input_img, decoded)
	autoencoder.summary()
	return autoencoder, encoded

global_autoencoder = None
global_encoder = None
global_input_img = None
def get_pretrained_conv_classifier(unlabeled_data, num_classes = 10, freeze_weights = True):
	global global_autoencoder
	global global_encoder
	global global_input_img

	if global_autoencoder is None:
		shape = unlabeled_data.shape[1:]
		input_img = Input(shape=shape)

		print('have to train autoencoder')
		autoencoder, encoder = get_conv_autoencoder_model(input_img, color_type=shape[0])

		autoencoder.compile(optimizer='rmsprop', 
							loss='binary_crossentropy',
							metrics=['mse'])

		weights_fn = 'weights_conv_auto_shape%s' % (str(shape))
		print('weights filename: %s' % weights_fn)

		if os.path.exists(weights_fn):
			print('loading weights from file')
			autoencoder.load_weights(weights_fn)
		else:
			print('no weights file, pretraining classifier')
			autoencoder.fit(unlabeled_data, unlabeled_data,
		                nb_epoch=5,
		                batch_size=256,
		                shuffle=True,
		                )

			autoencoder.save_weights(weights_fn, overwrite=True)

		global_autoencoder = autoencoder
		global_encoder = encoder
		global_input_img = input_img
	else:
		autoencoder = global_autoencoder
		encoder = global_encoder
		input_img = global_input_img

		print('already have trained autoencoder in memory')

	x = Flatten()(encoder)
	
	if 0:
		x = Dropout(0.5)(x)
		x = Dense(50, activation='sigmoid')(x)
		x = Dropout(0.5)(x)

	x = Dense(num_classes)(x)
	x = Activation('softmax')(x)

	classifier = Model(input_img, x)	


	for trained_layer, new_layer, _ in zip(autoencoder.layers, classifier.layers, range(9)):
		new_layer.set_weights(trained_layer.get_weights())
		print('copying weights on layer: %s' % str(trained_layer))
		if freeze_weights:		#freeze lower level weights
			new_layer.params = []
			new_layer.updates = []

	print('weights sum (1st conv): ', np.sum(classifier.layers[1].get_weights()[0]))
	print('weights sum (top dense): ', np.sum(classifier.layers[-2].get_weights()[0]))

		
	classifier.summary()

	optimizer = 'adadelta'
	#optimizer = Adam(lr=1e-3)
	#optimizer = RMSprop()
	classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return classifier

def test_mnist():	
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
	x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))


	classifier = get_pretrained_conv_classifier(x_train)	
	for _ in range(3):
		classifier.fit(x_train, np_utils.to_categorical(y_train),
				nb_epoch = 2,
				batch_size=256,
				shuffle=True,
				validation_data=(x_test, np_utils.to_categorical(y_test))
				)	

from run_keras_cv_drivers_v2 import *	
def test_state_farm(nfolds=13):


	 # input image dimensions
	img_rows, img_cols = 64, 64
	# color type: 1 - grey, 3 - rgb
	color_type_global = 1
	batch_size = 16
	nb_epoch = 50
	random_state = 51
	restore_from_last_checkpoint = 0

	train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global)
	test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

	all_input_data = np.concatenate([train_data, test_data])

	#model = create_model_v1(img_rows, img_cols, color_type_global)
	
	yfull_train = dict()
	yfull_test = []
	kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
	num_fold = 0
	sum_score = 0
	for train_drivers, test_drivers in kf:

		model = get_pretrained_conv_classifier(all_input_data)		
	
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

		kfold_weights_path = os.path.join('cache', 'weights_kfold_' + str(num_fold) + '.h5')
		if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
			callbacks = [
			    EarlyStopping(monitor='val_loss', patience=1, verbose=0),
			    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
			]
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
			      shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
			      callbacks=callbacks)


			print('weights sum (1st conv): ', np.sum(model.layers[1].get_weights()[0]))
			print('weights sum (top dense): ', np.sum(model.layers[-2].get_weights()[0]))

		if os.path.isfile(kfold_weights_path):
		    model.load_weights(kfold_weights_path)

		predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
		score = log_loss(Y_valid, predictions_valid)
		print('Score log_loss: ', score)
		sum_score += score*len(test_index)

		# Store valid predictions
		for i in range(len(test_index)):
			yfull_train[test_index[i]] = predictions_valid[i]

		# Store test predictions
		test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
		yfull_test.append(test_prediction)

	score = sum_score/len(train_data)
	print("Log_loss train independent avg: ", score)

	predictions_valid = get_validation_predictions(train_data, yfull_train)
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
	create_submission(test_res, test_id, info_string)
	save_useful_data(predictions_valid, train_id, model, info_string)



if __name__ == "__main__":	
	# test_mnist()

	test_state_farm()