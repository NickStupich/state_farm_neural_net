# -*- coding: utf-8 -*-

import numpy as np

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D
from keras.callbacks import EarlyStopping, ModelCheckpoint	

# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation


mean_pixel = [103.939, 116.779, 123.68] #vgg16 paper
#mean_pixel = [95.079, 96.925, 80.067]   #train data mean

np.random.seed(2016)
use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 3

# color_type = 1 - gray
# color_type = 3 - RGB

from vgg16_efficiency import get_trained_vgg16_model_2

def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    # mean_pixel = [103.939, 116.799, 123.68]
    # resized = resized.astype(np.float32, copy=False)

    # for c in range(3):
    #    resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized

def get_driver_data():
    dr = dict()
    path = os.path.join('driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('train',
                            'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers

def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id

def cache_data(data, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)

def read_model(index, cross=''):
    
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    print('reading model. weights name: %s' % weight_name)
    if 1:
        model = model_from_json(open(os.path.join('cache', json_name)).read())
    else:
        #model = vgg_std16_model(224, 224, 3)
        model = vgg_std16_model2(224, 224, 3)

    model.load_weights(os.path.join('cache', weight_name))
    return model

def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = \
        train_test_split(train, target,
                         test_size=test_size,
                         random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

def read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                              color_type=1, shuffle=True):

    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers),
                   cache_path)
    else:
        print('Restore train from cache: %s' % cache_path)
        (train_data, train_target, driver_id, unique_drivers) = \
            restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], color_type,
                                        img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    for c in range(3):
        train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    # train_data /= 255
    if shuffle:
        perm = permutation(len(train_target))
        train_data = train_data[perm]
        train_target = train_target[perm]
        
    print('Train shape:', train_data.shape)
    print('Train mean: ', np.mean(train_data, axis=1))
    return train_data, train_target, driver_id, unique_drivers

def read_and_normalize_test_data(img_rows=224, img_cols=224, color_type=1, index_range=None):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache: %s' % cache_path)
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)

    if index_range:
        test_data = test_data[index_range[0]:index_range[1]]
        test_id = test_id[index_range[0]:index_range[1]]

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], color_type,
                                      img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    for c in range(3):
        test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
    # test_data /= 255
    print('Test shape:', test_data.shape)
    print('Test mean: %s' % np.mean(test_data, axis=1))
    return test_data, test_id

def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()

def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index

def vgg_std16_model2(img_rows, img_cols, color_type):
    model = get_trained_vgg16_model_2(img_rows, img_cols, color_type, 10)
    #model.layers.pop()
    #model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    model.summary()
    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def cross_validation_train(nfolds=10, nb_epoch=10, modelStr='', img_rows = 224, img_cols = 224, batch_size=8, random_state=20):
    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                                  color_type_global, shuffle=False)

    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    
    for num_fold, (train_drivers, test_drivers) in enumerate(kf):
        model = vgg_std16_model2(img_rows, img_cols, color_type_global)

        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        print('Train drivers: ', unique_list_train)
        print('Test drivers: ', unique_list_valid)

        
        weights_path = 'vgg16_xval_models/fold%d.h5' % num_fold

        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', verbose=0))
        callbacks.append(ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0))
    
        
        model.fit(X_train, Y_train, 
                batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  verbose=1,
                  validation_data=(X_valid, Y_valid),
                  shuffle=True,
                  callbacks=callbacks)

        save_model(model, num_fold, modelStr)


def run_cross_validation(nfolds=10, nb_epoch=10, modelStr=''):

    img_rows, img_cols = 224, 224
    batch_size = 8
    random_state = 21

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

        split_test_data, split_ids = read_and_normalize_test_data(img_rows, img_cols, color_type_global, index_range = index_range)
        
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

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    create_submission(test_res, test_id, info_string)

def run_continuous_epochs(nb_epoch=20, val_split=0.1, modelStr=''):

    img_rows, img_cols = 224, 224
    batch_size = 16
    random_state = 20

    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                                  color_type_global, shuffle=True)

    
    for num_model in range(1000):

        print('Start KFold number {} from {}'.format(num_model, 1000))
        
        model = vgg_std16_model2(img_rows, img_cols, color_type_global)
        weights_path = 'vgg16_full_models/continuous_fold%d.h5' % num_model

        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', verbose=0))
        callbacks.append(ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0))
    
        
        model.fit(train_data, train_target, 
                batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  verbose=1,
                  validation_split=val_split, 
                  shuffle=True,
                  callbacks=callbacks)

        model.load_weights(weights_path)

        splits = 5
        n = 79726
        all_predictions = []
        all_test_ids = []
        for split in range(splits):
            index_range = [int(split*n/splits),int((split+1)*n/splits)]

            split_test_data, split_ids = read_and_normalize_test_data(img_rows, img_cols, color_type_global, index_range = index_range)
            all_test_ids += split_ids
            print('split test data shape: %s' % str(split_test_data.shape))

            predictions = model.predict(split_test_data, batch_size = batch_size, verbose=True)
            all_predictions += list(predictions)


        info_string = 'loss_' + modelStr \
                      + '_r_' + str(img_rows) \
                      + '_c_' + str(img_cols) \
                      + '_model_num_' + str(num_model)

        print('creating submission file %s' % info_string)

        create_submission(all_predictions, all_test_ids, info_string)

if __name__ == "__main__":
    # nfolds, nb_epoch, split
    run_cross_validation(2, 20, '_vgg_16_2x20')

    #run_continuous_epochs(modelStr = 'continuous_runs_2')

    # nb_epoch, split
    # run_one_fold_cross_validation(10, 0.1)

    # test_model_and_submit(1, 10, 'high_epoch')
