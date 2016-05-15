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
import os
import timeit

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA
from SdA import SdA
from utils import tile_raster_images_color

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
                                            corruption_level = 0.2,
                                            all_input_data = None,
                                            ):


    fn = '%s/encoded_train_layer%d_size%s_hidden%d.npy' % (output_folder, layer, str(img_shape), n_hidden)
    if os.path.exists(fn):
        result = np.load(fn)
        return result
    else:
        print('cached filename "%s" does not exist, re-running autoencoder' % fn)

    img_cols, img_rows, color_type_global = img_shape

    random_state = 51

    if all_input_data is None:

        train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global, one_hot_label_encoding = False)
        test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

        all_input_data = np.concatenate((train_data, test_data))
        print(all_input_data.shape)

        #all_input_data = all_input_data.reshape((all_input_data.shape[0], all_input_data.shape[1]*all_input_data.shape[2]*all_input_data.shape[3]))
        all_input_data = all_input_data.reshape((all_input_data.shape[0], -1), order='F')
        all_input_data = all_input_data[::5]    #TODO: not this...
    
    if n_hidden >= all_input_data.shape[1]:
        print('output size > input size, pca would give perfect reconstruction')
    elif 1: #print out a benchmark pca decomposition
        print('calculating pca mean squared error...')
        pca = PCA(n_components = n_hidden)
        train_components = pca.fit_transform(all_input_data)
        reconstruction = pca.inverse_transform(train_components)
        mse = mean_squared_error(all_input_data, reconstruction)
        print('pca mse (train): %s' % mse)


    all_input_data_shared = theano.shared(all_input_data, borrow=True)
    #all_input_data_shared = theano.tensor._shared(all_input_data, borrow=True)



    n_train_batches = all_input_data.shape[0] // batch_size

    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    print("Building model...")
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=all_input_data.shape[1], #img_rows*img_cols*color_type_global,
        n_hidden=n_hidden
    )

    cost, updates = da.get_cost_updates(
        corruption_level=corruption_level,
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

    train_sum_squared_err_da = theano.function(
            inputs = [index],
            outputs = da.training_mean_squared_reconstruction_error(),
            givens = {
                x: all_input_data_shared[index*batch_size:(index+1)*batch_size]
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
        sum_squared_error = 0
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
            sum_squared_error += train_sum_squared_err_da(batch_index)

        print('Training epoch %d, cost: %f  mse: %f' % 
            (epoch, np.mean(c), sum_squared_error / n_train_batches))
        #print('Training epoch %d, cost: %f' % (epoch, np.mean(c)))
        if save_filters:
            image = Image.fromarray(
                tile_raster_images_color(X=da.W.get_value(borrow=True).T,
                                   img_shape=img_shape, tile_shape=(10, 10),
                                   tile_spacing=(1, 1)))
            image.save('%s/filters_corruption_%d.png' % (output_folder, epoch))


    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The autoencoder in file ' +
           os.path.split(__file__)[1] +
           ' trained for %.2fm' % ((training_time) / 60.)), file=sys.stderr)

    get_encoded_vals = theano.function(
        inputs = [x],
        outputs = da.get_hidden_values(x),
    )
    print('made get_encoded_vals function')
    encoded_input_data = np.zeros((train_data.shape[0], n_hidden), dtype='float32')
    reshaped_train_data = train_data.reshape((train_data.shape[0], -1), order='F')
    for i in range(0, train_data.shape[0], batch_size):
        encoded_input_data[i:i+batch_size] = get_encoded_vals(reshaped_train_data[i:i+batch_size])

    np.save(fn, encoded_input_data)

    return encoded_input_data

def create_logistic_model(input_length):
    global printedSummary
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(input_length,)))

    model.add(Dense(100))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    if not printedSummary:
        model.summary()
        printedSummary = True

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
    return model


if __name__ == "__main__":
    output_folder = 'denoising_ae_preprocess'
    img_shape = (64, 48, 3)
    n_hidden = 500

    train_data = run_full_autoencoder_cross_validation(  batch_size=10, 
                                            epochs = 100, 
                                            learning_rate = 0.01,
                                            output_folder = output_folder,
                                            n_hidden = n_hidden,
                                            save_filters = True,
                                            layer = 0,
                                            img_shape = img_shape,
                                        )   

    #print('layer0 output type: %s   shape: %s' % (str(train_data_layer0.dtype), str(train_data_layer0.shape)))

    # train_data = run_full_autoencoder_cross_validation(  batch_size=10, 
    #                                     epochs = 100, 
    #                                     learning_rate = 0.01,
    #                                     output_folder = output_folder,
    #                                     n_hidden = n_hidden,
    #                                     save_filters = False,
    #                                     layer = 1,
    #                                     img_shape = img_shape,
    #                                     all_input_data = train_data_layer0
    #                                 )   


    nfolds = 13
    layer = 0
    nb_epoch = 100
    batch_size = 64
    random_state = 51
    fn = output_folder + '/' + 'encoded_train_layer%d_size%s_hidden%d.npy' % (layer, img_shape, n_hidden)

    img_cols, img_rows, color_type_global = img_shape

    _, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global, one_hot_label_encoding = True)
    
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

        # Store valid predictions
        # for i in range(len(test_index)):
        #     yfull_train[test_index[i]] = predictions_valid[i]

        # # Store test predictions
        # test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
        # yfull_test.append(test_prediction)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)
