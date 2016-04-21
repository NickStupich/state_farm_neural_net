from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras.regularizers import *
from keras.optimizers import SGD

def create_model_logReg(img_rows, img_cols):
    nb_classes = 10
    model = Sequential()

    # model.add(Dense(output_dim=nb_classes, input_dim=(img_rows*img_cols), init="glorot_uniform"))#, W_regularizer=l2(1E0)))
    model.add(Dense(output_dim=nb_classes))
    model.add(Dropout(0.5))

    # model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    sgd = SGD(lr=1E-3)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def create_model_conv1(img_rows, img_cols, isColor = 0):
    nb_classes = 10
    model = Sequential()

    # number of convolutional filters to use
    nb_filters = 16
    # size of pooling area for max pooling
    nb_pool = 3
    # convolution kernel size
    nb_conv = 3
    model = Sequential()

    colorDim = 3 if isColor else 1

    model.add(Reshape(input_shape=(img_rows*img_cols*colorDim,), target_shape = (colorDim, img_rows, img_cols)))
    #model.add(GaussianNoise(0.1, input_shape=(1, img_rows, img_cols)))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid'))
    #model.add(Dropout(0.5))

    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))

    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # model.add(Dense(32))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=2E-2, momentum=0.5, decay = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"], )

    #model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])

    return model
