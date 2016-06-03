from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation
from keras.optimizers import *
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import pylab
import cv2
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint	
from keras.optimizers import *
from sklearn.metrics import mean_squared_error, log_loss
# from run_keras_cv_drivers_v2 import *	
from pretrained_vgg16 import *
import pickle


printedSummary = False
def create_logistic_model(encoded_shape):
    global printedSummary
    model = Sequential()
    #model.add(Flatten(input_shape = (encoded_shape)))

    model.add(Dense(10, input_shape = (encoded_shape)))
    model.add(Activation('softmax'))

    if not printedSummary:
        model.summary()
        printedSummary = True

    #optimizer = SGD(lr=1e-2, momentum = 0.9, decay = 0.0)
    #optimizer = RMSprop()
    optimizer = Adam(lr=4e-1)
    #optimizer = Adadelta()
    print('using optimizer: %s' % str(optimizer))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

def create_mlp_model(encoded_shape, layers = [100]):
    global printedSummary
    model = Sequential()
    #model.add(Flatten(input_shape = (encoded_shape)))


    model.add(Dense(layers[0], input_shape = encoded_shape, activation='relu', init='he_normal'))

    for i, n in enumerate(layers[1:]):
        model.add(Dropout(0.5))
        model.add(Dense(n, activation='relu', init='he_normal'))

    model.add(Dense(10, input_shape = (encoded_shape)))
    model.add(Activation('softmax'))

    if not printedSummary:
        model.summary()
        printedSummary = True

    model.compile(Adam(lr=1e-2), loss='categorical_crossentropy')
    return model

def create_vgg16_dense_model(encoded_shape):
	#plan is to train all the dense layers at the top of vgg16, keeping convs constant
	pass

def get_trained_vgg16_model(img_rows, img_cols, color_type):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('vgg16_weights.h5')

    return model

def vgg_std16_encoder(img_rows, img_cols, color_type=1, include_connected_layers = True):
    
    model = get_trained_vgg16_model(img_rows, img_cols, color_type)

    # Code above loads pre-trained data and
    model.layers.pop()#softmax layer

    if not include_connected_layers:
    	for _ in range(4):
    		model.layers.pop()

    model.summary()

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def cross_validation_wth_encoder_no_finetune(img_shape, 
											nfolds=13, 
											do_test_predictions = False, 
											folder_name='folder_name', 
											model_build_func = create_logistic_model,
											retrain_single_model = True):
	nb_epoch = 2000
	batch_size = 128
	random_state = 51
	restore_from_last_checkpoint = 0
	img_rows, img_cols, color_type = img_shape

	#vgg16 includes dense layers?
	include_connected = True


	if do_test_predictions:

		test_fn = folder_name + '/vgg_encoded_test_connected-%s.pickle' % include_connected
		print('encoded data filename: %s' % test_fn)
		if os.path.exists(test_fn):
			encoded_test_data, test_id = pickle.load(open(test_fn, 'rb'))
		else:
			print('getting test data encoding')

			encoder = vgg_std16_encoder(img_shape[0], img_shape[1], img_shape[2], include_connected_layers = include_connected)

			splits = 10
			n = 79726
			test_id = []
			all_encoded_splits = []
			for split in range(splits):
				print('running split %d' % split)
				index_range = [int(split*n/splits),int((split+1)*n/splits)]
				print(index_range)

				split_test_data, split_ids = read_and_normalize_test_data(*img_shape, index_range = index_range)
				test_id += split_ids

				split_encoded = encoder.predict(split_test_data, batch_size = 8, verbose=True)
				all_encoded_splits.append(split_encoded)

			encoded_test_data = np.concatenate(all_encoded_splits)
			print('encoded test data shape: %s' % str(encoded_test_data.shape))

			#encoded_test_data = encoder.predict(test_data, batch_size = 8, verbose=True)
			pickle.dump((encoded_test_data, test_id), open(test_fn, 'wb'))

			encoder = None


	train_data, train_target, driver_id, unique_drivers = read_and_normalize_and_shuffle_train_data(*img_shape)

	fn = folder_name + '/vgg_encoded_connected-%s.npy' % include_connected
	print('encoded data filename: %s' % fn)
	if os.path.exists(fn):
		print('getting encoded data from file')
		encoded_train_data = np.load(fn)
	else:
		print('using vgg16 to encode data, then save to file')
		encoder = vgg_std16_encoder(img_shape[0], img_shape[1], img_shape[2], include_connected_layers = include_connected)

		batch = 8 if include_connected else 16
		encoded_train_data = encoder.predict(train_data, batch_size = batch, verbose=True)
		np.save(fn, encoded_train_data)
		encoder = None
	
	encoded_shape = encoded_train_data.shape
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
			model = model_build_func(encoded_shape[1:])

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
			callbacks = []
			callbacks.append(EarlyStopping(monitor='val_loss', patience=50, verbose=0))
			callbacks.append(ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0))
			
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
		for i in range(len(test_index)):
			yfull_train[test_index[i]] = predictions_valid[i]

		# Store test predictions
		if do_test_predictions:
			test_prediction = model.predict(encoded_test_data, batch_size=batch_size, verbose=1)
			yfull_test.append(test_prediction)

	score = sum_score/len(encoded_train_data)
	print("Log_loss train independent avg: ", score)

	# predictions_valid = get_validation_predictions(encoded_train_data, yfull_train)
	# score1 = log_loss(train_target, predictions_valid)
	# if abs(score1 - score) > 0.0001:
	#     print('Check error: {} != {}'.format(score, score1))

	print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
	info_string = 'loss_' + str(score) \
			+ '_r_' + str(img_rows) \
			+ '_c_' + str(img_cols) \
			+ '_folds_' + str(nfolds) \
			+ '_ep_' + str(nb_epoch)


	if do_test_predictions:
		# test_res = merge_several_folds_geom(yfull_test, nfolds)
		test_res = merge_several_folds_mean(yfull_test, nfolds)
		create_submission(test_res, test_id, info_string)

	#save_useful_data(predictions_valid, train_id, model, info_string)


def main():

	img_rows, img_cols = 224, 224
	color_type = 3
	input_shape = [img_rows, img_cols, color_type]

	folder_name = 'vgg16_encoder1'

	if not os.path.exists(folder_name):	os.mkdir(folder_name)

	# encoder = vgg_std16_encoder(img_rows, img_cols, color_type)

	# print('got encoder')


	#model_builder = create_logistic_model
	model_builder = create_mlp_model

	cross_validation_wth_encoder_no_finetune(input_shape, 
									nfolds=2, 
									do_test_predictions = True,
									folder_name=folder_name, 
									model_build_func = model_builder,
									retrain_single_model = False)

if __name__ == "__main__":
	main()
