import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics
import time
from shutil import copy2
import warnings
import random
warnings.filterwarnings("ignore")

import sys
import timeit

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA
from SdA import SdA
from utils import tile_raster_images

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize, imshow


try:
    import PIL.Image as Image
except ImportError:
    import Image

from run_keras_cv_drivers_v2 import *

def run_full_autoencoder_cross_validation(  batch_size=10, 
                                            epochs = 100, 
                                            learning_rate = 0.01,
                                            output_folder = 'denoising_ae_preprocess',
                                            n_hidden = 500,
                                            save_filters = True,
                                            layer = 0,
                                            img_shape = (64, 48, 1),
                                            ):
    # input image dimensions
    
    img_rows, img_cols, color_type_global = img_shape

    random_state = 51

    train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global, one_hot_label_encoding = False)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

    all_input_data = np.concatenate((train_data, test_data))
    print(all_input_data.shape)

    #all_input_data = all_input_data.reshape((all_input_data.shape[0], all_input_data.shape[1]*all_input_data.shape[2]*all_input_data.shape[3]))
    all_input_data = all_input_data.reshape((all_input_data.shape[0], -1), order='F')
    all_input_data = all_input_data[::5]    #TODO: not this...
    print(all_input_data.shape)


    if n_hidden < all_input_data.shape[1]:
        pca = PCA(n_components = n_hidden)
        train_components = pca.fit_transform(all_input_data)
        # print(pca.explained_variance_ratio_)
        # print(train_components.shape)
        reconstruction = pca.inverse_transform(train_components)
        # print(X_train_reconstruction.shape)
        # print(X_train_flat.shape)
        # print(X_train_flat[0,:20])
        # print(X_train_reconstruction[0, :20])
        mse = mean_squared_error(all_input_data, reconstruction)
        print('pca mse (train): %s' % mse)
    else:
        print('output size > input size, pca would give perfect reconstruction')

    all_input_data_shared = theano.shared(all_input_data, borrow=True)



    n_train_batches = all_input_data.shape[0] // batch_size

    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    print("Building model...")
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=img_rows*img_cols*color_type_global,
        n_hidden=n_hidden
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.1,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: all_input_data_shared[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_mse_da = theano.function(
            inputs = [],
            outputs = da.training_mean_squared_reconstruction_error(),
            givens = {
                x: all_input_data_shared
            }
        )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost: %f  mse: %f' % (epoch, np.mean(c), train_mse_da()))
        if save_filters:
            image = Image.fromarray(
                tile_raster_images(X=da.W.get_value(borrow=True).T,
                                   img_shape=(img_cols, img_rows), tile_shape=(10, 10),
                                   tile_spacing=(1, 1)))
            image.save('filters_corruption_%d.png' % epoch)


    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The no corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)

    get_encoded_vals = theano.function(
        inputs = [x],
        outputs = da.get_hidden_values(x),
    )

    #encoded_all_data = get_encoded_vals(all_input_data)
    #print('got encoded training data:')
    #print(encoded_training_data.shape)

    fn = 'encoded_train_layer%d_size%s_hidden%d' % (layer, str(img_shape), n_hidden)
    encoded_input_data = get_encoded_vals(train_data.reshape((train_data.shape[0], -1), order='F'))
    np.save(fn, encoded_input_data)

    os.chdir('../')

    return encoded_input_data

def create_logistic_model(input_length):
    global printedSummary
    model = Sequential()

    model.add(Dense(10, input_shape=(input_length,)))
    model.add(Activation('softmax'))

    if not printedSummary:
        model.summary()
        printedSummary = True

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
    return model


if __name__ == "__main__":
    output_folder = 'denoising_ae_preprocess'
    img_shape = (64, 48, 1)

    if 0:
        run_full_autoencoder_cross_validation(  batch_size=10, 
                                                epochs = 50, 
                                                learning_rate = 0.01,
                                                output_folder = output_folder,
                                                n_hidden = 500,
                                                save_filters = True,
                                                layer = 0,
                                                img_shape = img_shape,
                                            )   


    nfolds = 13
    layer = 0
    n_hidden = 500
    nb_epoch = 100
    batch_size = 64
    random_state = 51
    fn = output_folder + '/' + 'encoded_train_layer%d_size%s_hidden%d.npy' % (layer, img_shape, n_hidden)

    img_rows, img_cols, color_type_global = img_shape

    train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global, one_hot_label_encoding = True)
    train_data = np.load(fn)

    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    for train_drivers, test_drivers in kf:

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


        model = create_logistic_model(n_hidden)


        kfold_weights_path = os.path.join('cache', 'weights_kfold_' + str(num_fold) + '.h5')
        #if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
     
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=1, verbose=0),
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        # for i in range(len(test_index)):
        #     yfull_train[test_index[i]] = predictions_valid[i]

        # # Store test predictions
        # test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
        # yfull_test.append(test_prediction)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)
