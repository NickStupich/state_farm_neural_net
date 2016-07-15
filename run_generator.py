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

from train_data_generator import driver_split_data_generator

#models
import vgg16_efficiency
import resnet50

color_type_global = 3

img_rows, img_cols = 224, 224

horizontal_flip = False
rotation_range = 20
width_shift_range = 0.05
height_shift_range = 0.05
shear_range = 10.0 / (180.0 / np.pi)
zoom_range = 0.1
channel_shift_range=10.

reset_weights_each_fold = False

samples_per_epoch = 4800

#learning_rates = [2e-6, 2e-7, 2e-8]
learning_rates = [1e-3]

batch_size = 48
random_state = 30

driver_split=True
num_folds = 4
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
                                        color_type, shuffle=False, transform=False)

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

        if reset_weights_each_fold or model is None or len(learning_rates) > 1:
            if reset_weights_each_fold or model is None:
                model = get_model()
            optimizer = get_optimizer(learning_rates[0])
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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

def generator_test_predict(model, test_data, batch_size=32, num_samples=4, generator_batch_size = 2**12):
    test_ids = np.arange(len(test_data)) #to put things back together after

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

    test_flow = test_datagen.flow(test_data, test_ids, batch_size=generator_batch_size)
    num_generator_batches = math.ceil(len(test_data) / generator_batch_size)

    for n_sample in range(num_samples):
        print('\nrunning random test sample %d / %d' % (n_sample, num_samples))
        for i, (batch_data, batch_ids) in enumerate(test_flow):
            preds = model.predict(batch_data, batch_size=batch_size, verbose=True)
            all_predictions[batch_ids, n_sample] = preds
            print('%d / %s' % (i, num_generator_batches))
            if i == (num_generator_batches-1):
                break

    predictions = np.mean(all_predictions, axis=1)
    print('different predictions average std dev: %s' % np.mean(np.std(all_predictions, axis=1, ddof=1)))
    print(all_predictions.shape)
    print(predictions.shape)

    return predictions


def generator_test_predict2(model, test_data, batch_size=32, num_samples=4):
    test_ids = np.arange(len(test_data)) #to put things back together after

    all_predictions = np.zeros((num_samples, len(test_data), 10))

    test_augment_range = 0.5

    test_datagen = ImageDataGenerator(
                    rotation_range=rotation_range * test_augment_range,
                    width_shift_range=width_shift_range * test_augment_range,
                    height_shift_range=height_shift_range * test_augment_range,
                    shear_range = shear_range * test_augment_range,
                    zoom_range = zoom_range * test_augment_range,
        channel_shift_range=channel_shift_range * test_augment_range,   
    )

    test_flow = test_datagen.flow(test_data, test_ids, batch_size=batch_size)

    for n_sample in range(num_samples):
        print('random test sample %d / %d' % (n_sample, num_samples))
        preds = model.predict_generator(test_flow, len(test_data))

        all_predictions[n_sample] = preds

    predictions = np.mean(all_predictions, axis=0)
    print('different predictions average std dev: %s' % np.mean(np.std(all_predictions, axis=0, ddof=1)))
    print(all_predictions.shape)
    print(predictions.shape)

    return predictions

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
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
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
                predictions = model.predict(split_test_data, batch_size = batch_size, verbose=True)
            else:
                #predictions = generator_test_predict(model, split_test_data, batch_size=batch_size, num_samples=num_test_samples)
                predictions = generator_test_predict2(model, split_test_data, batch_size=batch_size, num_samples=num_test_samples)
 
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
    modelStr = 'run_gen_%s_num_test_samples_%s' % (model_name, num_test_samples)
    num_test_samples = 2
    run_cross_validation(num_folds, num_epochs, modelStr, num_test_samples = num_test_samples)

    # num_test_samples = 3
    # modelStr = 'run_gen_%s_num_test_samples_%s' % (model_name, num_test_samples)    
    # run_cross_validation(num_folds, num_epochs, modelStr, num_test_samples = num_test_samples)

if __name__ == "__main__":
	main()
