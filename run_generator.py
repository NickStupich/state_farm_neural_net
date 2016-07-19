import numpy as np
import os
import datetime
import pandas as pd
import pickle
import sys
import math
import pylab
import cv2

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint	
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import pretrained_vgg16
from numpy.random import permutation

from train_data_generator import driver_split_data_generator, test_data_generator
from pretrained_vgg16 import read_and_normalize_and_shuffle_train_data, copy_selected_drivers
from average_submissions import average_submissions

#models
import vgg16_efficiency
import resnet50

color_type_global = 3

img_rows, img_cols = 224, 224

horizontal_flip = False
rotation_range = 30
width_shift_range = 0.1
height_shift_range = 0.1
shear_range_deg = 10.0
shear_range = shear_range_deg / (180.0 / np.pi)
zoom_range = 0.1
channel_shift_range=10.

augment_specs = '_'.join(map(str, [rotation_range, width_shift_range, height_shift_range, shear_range_deg, zoom_range, channel_shift_range]))
reset_weights_each_fold = True

samples_per_epoch = 4800

#learning_rates = [2e-6, 2e-7, 2e-8]
learning_rates = [1e-3]

batch_size = 48
test_batch_size = batch_size
random_state = 30

driver_split=False
num_folds = 20
num_epochs = 20
num_test_samples = 1
patience=2

#model_name = 'resnet50' 
#get_model = resnet50.resnet50

#get_model = resnet50.resnet_small
#model_name = 'resnet_small'

# get_model = resnet50.resnet_tiny
#model_name = 'resnet_tiny'

get_model = vgg16_efficiency.get_trained_vgg16_model_2
model_name = 'vgg16'

# get_model = vgg16_efficiency.trained_vgg16_no_fc
# model_name = 'vgg16_nofc'

# get_model = vgg16_efficiency.trained_vgg16_average_1x1
# model_name = 'vgg16_avg_1x1'

get_optimizer = lambda lr: SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
#get_optimizer = lambda lr: Adam()

def get_callbacks(weights_path):
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, verbose=0))
    callbacks.append(ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0))
    return callbacks

def cross_validation_train(nfolds=10, nb_epoch=10, modelStr='', img_rows = 224, img_cols = 224, batch_size=8, random_state=20, driver_split=True):
    model = None

    if driver_split:
        data_iterator = driver_split_data_generator(nfolds, img_rows, img_cols, color_type_global, random_state)
    else:
        train_data, train_target, driver_id, unique_drivers = read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                        color_type_global, shuffle=False, transform=False)

    for num_fold in range(nfolds):
        if driver_split:
            data_provider = next(data_iterator)

        model_path = os.path.join('cache', 'model_weights' + str(num_fold) + modelStr + '.h5')
        if os.path.isfile(model_path):
            print('already trained this fold, skipping...')
            continue

        print('Start KFold number {} from {}'.format(num_fold, nfolds))

        weights_path = 'generator_xval_models/fold%d.h5' % num_fold

        if driver_split:
            (X_train, Y_train, X_valid, Y_valid) = data_provider()

        if model is None:
            print('creating new model')
            model = get_model()
            optimizer = get_optimizer(learning_rates[0])
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        elif len(learning_rates) > 1:
            print('creating new optimizer and recompiling')
            optimizer = get_optimizer(learning_rates[0])
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if reset_weights_each_fold:
            print('resetting model weights to imagenet pretrain')
            vgg16_efficiency.set_vgg16_model_2_weights(model, set_last_layer = False)


        train_datagen = ImageDataGenerator(
                        rotation_range=rotation_range,
                        width_shift_range=width_shift_range,
                        height_shift_range=height_shift_range,
                        shear_range = shear_range,
                        zoom_range = zoom_range,
			channel_shift_range=channel_shift_range,	
		)

        if driver_split:
            perm = permutation(len(X_train))    
            train_flow = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

            model.fit_generator(train_flow,
             	samples_per_epoch = samples_per_epoch if samples_per_epoch > 0 else len(X_train),
    			 nb_epoch=nb_epoch,
    			 validation_data=(X_valid, Y_valid),
    			 callbacks=get_callbacks(weights_path),
    			 nb_val_samples = len(X_valid))

            model.load_weights(weights_path)


            for learning_rate in learning_rates[1:]:
                print('dropping learning rate to %s' % learning_rate)
                
                optimizer = get_optimizer(learning_rates[0])
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                model.fit_generator(train_flow,
                 samples_per_epoch = samples_per_epoch if samples_per_epoch > 0 else len(X_train),
                 nb_epoch=nb_epoch,
                 validation_data=(X_valid, Y_valid),
                 callbacks=get_callbacks(weights_path),
                 nb_val_samples = len(X_valid))

                model.load_weights(weights_path)

        else:            
            perm = permutation(len(train_data))    

            n_train = int(len(train_data)*0.9)
            n_valid = len(train_data) - n_train
            train_flow = train_datagen.flow(train_data[perm[:n_train]], train_target[perm[:n_train]], batch_size=batch_size)
            X_valid = train_data[perm[n_train:]]
            Y_valid = train_target[perm[n_train:]]

            model.fit_generator(train_flow,
                samples_per_epoch = samples_per_epoch if samples_per_epoch > 0 else n_train,
                 nb_epoch=nb_epoch,
                 validation_data=(X_valid, Y_valid),
                 callbacks=get_callbacks(weights_path),
                 nb_val_samples = n_valid)

            model.load_weights(weights_path)

            for learning_rate in learning_rates[1:]:
                print('dropping learning rate to %s' % learning_rate)
                optimizer = get_optimizer(learning_rates[0])
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                model.fit_generator(train_flow,
                 samples_per_epoch = samples_per_epoch if samples_per_epoch > 0 else n_train,
                 nb_epoch=nb_epoch,
                 validation_data=(X_valid, Y_valid),
                 callbacks=get_callbacks(weights_path),
                 nb_val_samples = n_valid)

                model.load_weights(weights_path)

        pretrained_vgg16.save_model(model, num_fold, modelStr)

def generator_test_predict2(model, test_data, batch_size=32, num_samples=4, random_state = 40):
    all_predictions = np.zeros((len(test_data), num_samples, 10))

    test_augment_range = 0.5

    test_datagen = ImageDataGenerator(
                    rotation_range=rotation_range * test_augment_range,
                    width_shift_range=width_shift_range * test_augment_range,
                    height_shift_range=height_shift_range * test_augment_range,
                    shear_range = shear_range * test_augment_range,
                    zoom_range = zoom_range * test_augment_range,
        channel_shift_range=channel_shift_range * test_augment_range,   
    )

    test_flow = test_datagen.flow(test_data, batch_size=batch_size, shuffle=False, seed=random_state)
    for n_sample in range(num_samples):
        test_flow.reset() #reset back to 0 (otherwise outputs are staggered?)
        preds = model.predict_generator(test_flow, len(test_data))

        print('random test sample %d / %d. shape: %s' % (n_sample, num_samples, str(preds.shape)))
        all_predictions[:, n_sample] = preds

    predictions = np.mean(all_predictions, axis=1)
    return predictions

def run_cross_validation2(nfolds=10, nb_epoch=10, modelStr='', num_test_samples=10):
    if driver_split:
        modelStr += '_driverSplit'
    else:
        modelStr += '_randomSplit'

    if 0:
        cross_validation_train(nfolds, nb_epoch, modelStr, img_rows, img_cols, batch_size, random_state, driver_split = driver_split)

    print('Start testing............')

    models = []

    folder = 'subm2/predictions_' + modelStr    
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    all_filenames = []
    for index in range(nfolds):
        filename = folder + 'fold_' + str(index) + 'test_samples_' + str(num_test_samples) + '.csv'
        all_filenames.append(filename)

        if os.path.exists(filename):
            print('file exists, skipping')
            continue

        model = pretrained_vgg16.read_model(index, modelStr)

        model_predictions = np.zeros((79726, 10))
        test_ids = []

        data_index = 0
        for test_fold, split_test_data_provider in enumerate(test_data_generator()):
            split_test_data, split_test_ids = split_test_data_provider()
            test_ids += list(split_test_ids)

            if num_test_samples == 1:
                predictions = model.predict(split_test_data, batch_size = test_batch_size, verbose=True)
            else:
                predictions = generator_test_predict2(model, split_test_data, batch_size=test_batch_size, num_samples=num_test_samples)
 
            model_predictions[data_index:data_index + len(predictions)] = predictions
            data_index += len(predictions)

        test_ids = np.array(test_ids)

        create_submission(model_predictions, test_ids, filename)

    average_filename = folder + 'folds_0-' + str(nfolds) + 'test_samples_' + str(num_test_samples) + '.csv'

    average_submissions(all_filenames, average_filename)


def create_submission(predictions, test_id, filename):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1.to_csv(filename, index=False)

def run_cross_validation(nfolds=10, nb_epoch=10, modelStr='', num_test_samples=10):
    if driver_split:
        modelStr += '_driverSplit'
    else:
        modelStr += '_randomSplit'

    if 1:
        cross_validation_train(nfolds, nb_epoch, modelStr, img_rows, img_cols, batch_size, random_state, driver_split = driver_split)

    print('Start testing............')

    yfull_test = np.zeros((nfolds, 79726, 10))
    print('yfull_test shape: %s' % str(yfull_test.shape))

    models = []
    for index in range(nfolds):
        model = pretrained_vgg16.read_model(index, modelStr)
        # model.compile(optimizer='sgd', loss='categorical_crossentropy')
        models.append(model)

    print('loaded %d models' % len(models))

    splits = 5
    n = 79726
    test_ids = []
    for split in range(splits):
        print('running split %d' % split)
        index_range = [int(split*n/splits),int((split+1)*n/splits)]
        print(index_range)

        split_test_data, split_ids = pretrained_vgg16.read_and_normalize_test_data(img_rows, img_cols, color_type_global, index_range = index_range, transform=False)
        test_ids += split_ids
        print('test ids array shape: %s' % str(len(test_ids)))

        print('split test data shape: %s' % str(split_test_data.shape))

        for index in range(nfolds):
            model = models[index]
    
            if num_test_samples == 1:
                predictions = model.predict(split_test_data, batch_size = test_batch_size, verbose=True)
            else:
                predictions = generator_test_predict2(model, split_test_data, batch_size=test_batch_size, num_samples=num_test_samples)
 
            print('predictions shape: %s' % str(predictions.shape))
            yfull_test[index, index_range[0]:index_range[1]] = predictions

    if num_test_samples > 1:
        modelStr += '_testaugment%d' % num_test_samples

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nfolds) \
                  + '_ep_' + str(nb_epoch)
    test_ids = np.array(test_ids)

    test_res = pretrained_vgg16.merge_several_folds_mean(yfull_test, nfolds)
    pretrained_vgg16.create_submission(test_res, test_ids, info_string)

def main():
    global num_test_samples
    modelStr = 'run_gen_%s_%s' % (model_name, augment_specs)
    print('modelstr: %s ' % modelStr)
    # run_cross_validation(num_folds, num_epochs, modelStr, num_test_samples = num_test_samples)
    run_cross_validation2(num_folds, num_epochs, modelStr, num_test_samples = num_test_samples)

if __name__ == "__main__":
	main()
