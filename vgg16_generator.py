import numpy as np
import os
import datetime
import pandas as pd
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint	
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import pretrained_vgg16

color_type_global = 3
img_rows, img_cols = 224, 224

horizontal_flip = True
rotation_range = 20
width_shift_range = 0.2
height_shift_range = 0.2
samples_per_epoch = 5000


def get_cached_train_data():
	fn_data = 'cache/cached_224x224x3_train_data.npy'
	fn_other = 'cache/cache_224x224x3_train_other.pickle'

	if os.path.isfile(fn_data) and os.path.isfile(fn_other):
		print('loading from cache')
		result0 = np.load(fn_data)
		result1 = pickle.load(open(fn_other, 'rb'))
		result = (result0, *result1)
	else:
		print('not loading from cache')
		result = pretrained_vgg16.read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                                  color_type_global, shuffle=False, transform=False)

		np.save(fn_data, result[0])
		pickle.dump(result[1:], open(fn_other, 'wb'))

	print('done loading train data')
	return result

def cross_validation_train(nfolds=10, nb_epoch=10, modelStr='', img_rows = 224, img_cols = 224, batch_size=8, random_state=20):
    train_data, train_target, driver_id, unique_drivers = get_cached_train_data()

    # train_data, train_target, driver_id, unique_drivers = \
    #     pretrained_vgg16.read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
    #                                               color_type_global, shuffle=False, transform=False)

    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    
    for num_fold, (train_drivers, test_drivers) in enumerate(kf):

        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = pretrained_vgg16.copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = pretrained_vgg16.copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        print('Train drivers: ', unique_list_train)
        print('Test drivers: ', unique_list_valid)

        weights_path = 'vgg16_generator_xval_models/fold%d.h5' % num_fold

        model = pretrained_vgg16.vgg_std16_model2(img_rows, img_cols, color_type_global)


        train_datagen = ImageDataGenerator(
			rotation_range=rotation_range,
			width_shift_range=width_shift_range,
			height_shift_range=height_shift_range,
			horizontal_flip=horizontal_flip)

        train_datagen.fit(X_train)

        train_flow = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

        valid_datagen = ImageDataGenerator(
			rotation_range=rotation_range,
			width_shift_range=width_shift_range,
			height_shift_range=height_shift_range,
			horizontal_flip=horizontal_flip)

        valid_datagen.fit(X_valid)

        valid_flow = train_datagen.flow(X_valid, Y_valid, batch_size=batch_size)

        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', verbose=0))
        callbacks.append(ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0))

   #      model.fit_generator(train_flow,
   #      	samples_per_epoch = samples_per_epoch if samples_per_epoch > 0 else len(X_train),
			# nb_epoch=nb_epoch,
			# validation_data=valid_flow,
			# callbacks=callbacks,
			# nb_val_samples = len(X_valid))


        model.fit_generator(train_flow,
        	samples_per_epoch = samples_per_epoch if samples_per_epoch > 0 else len(X_train),
			nb_epoch=nb_epoch,
            validation_data=(X_valid, Y_valid),	
			callbacks=callbacks,
			nb_val_samples = len(X_valid))

        #reload the best epoch
        model.load_weights(weights_path)

        save_model(model, num_fold, modelStr)

def run_cross_validation(nfolds=10, nb_epoch=10, modelStr=''):

    batch_size = 32
    random_state = 20

    if 1:
        cross_validation_train(nfolds, nb_epoch, modelStr, img_rows, img_cols, batch_size, random_state)

    print('Start testing............')

    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) +
            '_c_' + str(img_cols) + '_t_' +
            str(color_type_global) + '.dat')
    _, test_id = restore_data(cache_path)


    yfull_test = np.zeros((nfolds, len(test_id), 10))
    print('yfull_test shape: %s' % str(yfull_test.shape))

    models = []
    for index in range(nfolds):
        model = read_model(index, modelStr)
        model.summary()
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        models.append(model)

    print('loaded %d models' % len(models))

    splits = 5
    n = 79726
    for split in range(splits):
        print('running split %d' % split)
        index_range = [int(split*n/splits),int((split+1)*n/splits)]
        print(index_range)

        split_test_data, split_ids = pretrained_vgg16.read_and_normalize_test_data(img_rows, img_cols, color_type_global, index_range = index_range, transform=False)
        
        print('split test data shape: %s' % str(split_test_data.shape))

        for index in range(nfolds):
            model = models[index]
            predictions = model.predict(split_test_data, batch_size = batch_size, verbose=True)
            print('predictions shape: %s' % str(predictions.shape))
            yfull_test[index, index_range[0]:index_range[1]] = predictions


    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nfolds) \
                  + '_ep_' + str(nb_epoch)

    test_res = pretrained_vgg16.merge_several_folds_mean(yfull_test, nfolds)
    pretrained_vgg16.create_submission(test_res, test_id, info_string)


def main():
    generator_specs = ''
    run_cross_validation(4, 20, '_vgg_16_generator_%s')

if __name__ == "__main__":
	main()