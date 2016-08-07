import numpy as np
import matplotlib.pylab as plt

import os

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

import theano.tensor.nnet.abstract_conv as absconv

from imagenet_utils import decode_predictions, preprocess_input
import vgg16

def VGG_CAM(nb_classes = 10, num_input_channels = 1024, input_img_shape = (3, 224, 224)):

    input_img = Input(input_img_shape)
    vgg16_base = vgg16.VGG16(include_top=False, input_tensor = input_img)

    flatten_dims = (vgg16_base.layers[-2].output_shape[2:])

    x = vgg16_base.layers[-2].output

    x = Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same", name='final_conv')(x)
    x = AveragePooling2D(flatten_dims)(x)
    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(input_img, x)
    model.name = "VGGCAM"

    return model

get_cmap = None
def get_classmap(model, X, nb_classes, batch_size = 1):
    global get_cmap

    if get_cmap is None:
        height = X.shape[2]
        width = X.shape[3]

        final_conv_layer = model.get_layer('final_conv')

        upscale_ratio = int(width / final_conv_layer.output_shape[-1])

        inc = model.layers[0].input
        conv6 = final_conv_layer.output
        num_input_channels = final_conv_layer.get_weights()[0].shape[0]

        conv6_resized = absconv.bilinear_upsampling(conv6, upscale_ratio,
                                                    batch_size=batch_size,
                                                    num_input_channels=num_input_channels)
        WT = model.layers[-1].W.T
        conv6_resized = K.reshape(conv6_resized, (-1, num_input_channels, height * width))
        classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, height, width))
        get_cmap = K.function([inc], classmap)

    return get_cmap([X])

def plot_classmap(model, img):

    nb_classes = model.layers[-1].get_weights()[1].shape[0]
    input_shape = model.layers[0].output_shape[2:]

    if isinstance(img, str):
        img = image.load_img(img, target_size=input_shape)
        img = image.img_to_array(img)
        x = np.expand_dims(np.copy(img), axis=0)
        x = preprocess_input(x)
    else:
        x = np.expand_dims(img, axis=0)

    batch_size = 1

    classmap = get_classmap(model,
                            x,
                            nb_classes)

    if 0:
        plt.subplot(2, 1, 1)
        plt.imshow(np.transpose(img, (1, 2, 0)).astype('uint8'))
        plt.subplot(2, 1, 2)
        label = np.argmax(model.predict(x), axis=0)[0]

        plt.imshow(classmap[0, label, :, :],
                   cmap="jet",
                   alpha=1.0,
                   interpolation='nearest')
        plt.title(str(label))
        plt.show()
    else:
        label = np.argmax(model.predict(x), axis=1)[0]
        plt.imshow(np.transpose(img, (1, 2, 0)).astype('uint8'))
        plt.imshow(classmap[0, label, :, :],
                   cmap="jet",
                   alpha=0.5,
                   interpolation='nearest')
        plt.title(str(label))
        plt.show()

def cifar_test():

    from keras.datasets import cifar10
    from keras.utils import np_utils

    nb_epoch = 5
    batch_size = 128

    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train_preprocessed = preprocess_input(X_train.astype('float32'))
    X_test_preprocessed = preprocess_input(X_test.astype('float32'))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = VGG_CAM(10, input_img_shape = X_train.shape[1:])
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])

    model_path = 'cifar10_models/epoch_{epoch:03d}.h5'

    if os.path.exists(model_path.format(epoch = nb_epoch-1)):
        model.load_weights(model_path.format(epoch = nb_epoch-1))
    else:
        callbacks = [
                ModelCheckpoint(model_path, save_best_only=False, verbose=0),
            ]

        model.fit(X_train_preprocessed, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  callbacks = callbacks,
                  validation_data=(X_test_preprocessed, Y_test),
                  shuffle=True)

    for i in range(100):
        plot_classmap(model, X_test[i])

def state_farm_test():

    nb_epoch = 7
    batch_size = 48

    nb_classes = 10
    nfolds = 4
    random_state = 30

    color_type = 3
    img_rows = 224
    img_cols = 224

    model = VGG_CAM(10, input_img_shape = (color_type, img_rows, img_cols))
    optimizer = SGD(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    model_path = 'statefarm_models/epoch_{epoch:03d}.h5'

    if os.path.exists(model_path.format(epoch = nb_epoch-1)):
        model.load_weights(model_path.format(epoch = nb_epoch-1))
    else:
        from train_data_generator import driver_split_data_generator

        data_iterator = driver_split_data_generator(nfolds, img_rows, img_cols, color_type, random_state)
        (X_train, Y_train, X_valid, Y_valid) = next(data_iterator)()

        callbacks = [
                ModelCheckpoint(model_path, save_best_only=False, verbose=0),
            ]

        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  callbacks = callbacks,
                  validation_data=(X_valid, Y_valid),
                  shuffle=True)

    plot_folder = 'train/c3/'
    # plot_folder = 'test/'
    for fn in os.listdir(plot_folder):
        plot_classmap(model, plot_folder + fn)

if __name__ == "__main__":
    # cifar_test()
    state_farm_test()
