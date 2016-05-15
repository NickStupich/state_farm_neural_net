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

try:
    import PIL.Image as Image
except ImportError:
    import Image

from run_keras_cv_drivers_v2 import *

def run_full_autoencoder_cross_validation(  nfolds=10, 
                                            batch_size=10, 
                                            pretraining_epochs = 100, 
                                            pretrain_lr = 0.001,
                                            finetune_lr = 0.5,
                                            training_epochs = 1000,
                                            ):
    # input image dimensions
    img_rows, img_cols = 64, 64
    # color type: 1 - grey, 3 - rgb
    color_type_global = 1
    batch_size = 64
    nb_epoch = 50
    random_state = 51
    restore_from_last_checkpoint = 0

    train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global, one_hot_label_encoding = False)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

    all_input_data = np.concatenate((train_data, test_data))
    print(all_input_data.shape)

    #all_input_data = all_input_data.reshape((all_input_data.shape[0], all_input_data.shape[1]*all_input_data.shape[2]*all_input_data.shape[3]))
    all_input_data = all_input_data.reshape((all_input_data.shape[0], -1))
    all_input_data = all_input_data[::5]    #TODO: not this...
    print(all_input_data.shape)

    all_input_data = theano.shared(all_input_data, borrow=True)
    # all_input_data = theano.tensor._shared(all_input_data, borrow=True)

    n_train_batches = all_input_data.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    numpy_rng = np.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=img_rows*img_cols*color_type_global,
        hidden_layers_sizes=[100],
        n_outs=10
    )

    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=all_input_data,
                                                batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [0.1]
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c)))

            if i == 0 and (epoch % 100 == 0) or (epoch < 100 and epoch % 10 == 0) or (epoch < 10):
                image = Image.fromarray(
                    tile_raster_images(X=sda.dA_layers[i].W.get_value(borrow=True).T,
                                   img_shape=(img_cols, img_rows, color_type_global) if color_type_global == 3 else (img_cols, img_rows),
                                   tile_shape=(10, 10),
                                   tile_spacing=(1, 1)))

                image.save('SdA_state_farm/all_data_filters_corruption%d_epoch%d.png' % (i, epoch))


    end_time = timeit.default_timer()

    print(('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    #release gpu memory
    all_input_data.set_value([[]])


    #supervised top layer training

    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    for train_drivers, test_drivers in kf:
        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

        X_train = X_train.reshape((X_train.shape[0], -1))
        X_valid = X_valid.reshape((X_valid.shape[0], -1))

        print('X_train.shape: %s' % str(X_train.shape))
        print('Y_train.shape: %s' % str(Y_train.shape))
        print('X_valid.shape: %s' % str(X_valid.shape))
        print('Y_valid.shape: %s' % str(Y_valid.shape))
        print(X_train.dtype, Y_train.dtype, X_valid.dtype, Y_valid.dtype)


        datasets = (
            (theano.shared(X_train, borrow=True), theano.shared(Y_train, borrow=True)), 
            (theano.shared(X_valid, borrow=True), theano.shared(Y_valid, borrow=True)), 
            (theano.shared(X_valid, borrow=True), theano.shared(Y_valid, borrow=True))
                )

        print('... getting the finetuning functions')
        
        train_fn, validate_model, test_model = sda.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr
        )

        print('... finetunning the model')
        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0

        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = test_model()
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%, '
                'on iteration %i, '
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print(('The training code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)



    # model = create_model_v1(img_rows, img_cols, color_type_global)

    # yfull_train = dict()
    # yfull_test = []
    # kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    # num_fold = 0
    # sum_score = 0
    # for train_drivers, test_drivers in kf:

    #     unique_list_train = [unique_drivers[i] for i in train_drivers]
    #     X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    #     unique_list_valid = [unique_drivers[i] for i in test_drivers]
    #     X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

    #     num_fold += 1
    #     print('Start KFold number {} from {}'.format(num_fold, nfolds))
    #     print('Split train: ', len(X_train), len(Y_train))
    #     print('Split valid: ', len(X_valid), len(Y_valid))
    #     print('Train drivers: ', unique_list_train)
    #     print('Test drivers: ', unique_list_valid)

    #     kfold_weights_path = os.path.join('cache', 'weights_kfold_' + str(num_fold) + '.h5')
    #     if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
    #         callbacks = [
    #             EarlyStopping(monitor='val_loss', patience=1, verbose=0),
    #             ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    #         ]
    #         model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #               shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
    #               callbacks=callbacks)
    #     if os.path.isfile(kfold_weights_path):
    #         model.load_weights(kfold_weights_path)

    #     # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
    #     # print('Score log_loss: ', score[0])

    #     predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    #     score = log_loss(Y_valid, predictions_valid)
    #     print('Score log_loss: ', score)
    #     sum_score += score*len(test_index)

    #     # Store valid predictions
    #     for i in range(len(test_index)):
    #         yfull_train[test_index[i]] = predictions_valid[i]

    #     # Store test predictions
    #     test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
    #     yfull_test.append(test_prediction)

    # score = sum_score/len(train_data)
    # print("Log_loss train independent avg: ", score)

    # predictions_valid = get_validation_predictions(train_data, yfull_train)
    # score1 = log_loss(train_target, predictions_valid)
    # if abs(score1 - score) > 0.0001:
    #     print('Check error: {} != {}'.format(score, score1))

    # print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
    # info_string = 'loss_' + str(score) \
    #                 + '_r_' + str(img_rows) \
    #                 + '_c_' + str(img_cols) \
    #                 + '_folds_' + str(nfolds) \
    #                 + '_ep_' + str(nb_epoch)

    # test_res = merge_several_folds_mean(yfull_test, nfolds)
    # # test_res = merge_several_folds_geom(yfull_test, nfolds)
    # create_submission(test_res, test_id, info_string)
    # save_useful_data(predictions_valid, train_id, model, info_string)

if __name__ == "__main__":
    run_full_autoencoder_cross_validation(nfolds=13)