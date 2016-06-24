'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import functools

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers import Flatten, Dropout, Activation, Reshape, Merge
from keras.utils import np_utils
from keras.optimizers import *
from keras.layers.noise import GaussianNoise

from sklearn import preprocessing
from sklearn.cross_validation import KFold

import run_keras_cv_drivers_v2

class VAEContainer(object):
    def __init__(self, 
                encoder_layer, 
                decoder_layers,
                name
                ):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.name = name

def categorical_to_dense(labels):
    result = np.zeros((len(labels)), dtype='int8')
    for i, label in enumerate(labels):
        result[i] = np.argmax(label)

    return result

batch_size = 16
latent_dim = 1000
epsilon_std = 0.01
nb_epoch = 173
num_classes = 10

# input_shape = (3, 64, 64)
input_shape = (3, 128, 128)
color_type, img_rows, img_cols = input_shape

add_derivative = False

if 0:   #single conv layer + dense
    conv_size = 5
    intermediate_dim = 128
    model_name = 'conv%s_dense%d' % ('(%d_%d)' % (8, conv_size), intermediate_dim)
    encode_layers = [
        Reshape((1, 28, 28)),
        Convolution2D(8, conv_size, conv_size, activation='relu', border_mode='same'),
        MaxPooling2D((conv_size-1, conv_size-1)),
        Flatten(),
        Dense(intermediate_dim, activation='relu'),
    ]

    decode_layers = [
        Dense(intermediate_dim, activation='relu'),
        Dense((int(8*28*28 / ((conv_size-1)**2))), activation='sigmoid'),
        Reshape((8, int(28 / (conv_size-1)), int(28 / (conv_size-1)))),
        UpSampling2D((conv_size-1, conv_size-1)),
        Convolution2D(1, conv_size, conv_size, activation='sigmoid', border_mode='same'),
        Flatten()
    ]

    vae_specs = VAEContainer(encode_layers, decode_layers, model_name)

elif 0:
    conv_size = 3
    conv_name = ('(8_3)(16_3)(32_3)')
    model_name = 'conv%s' % (conv_name)

    encode_layers = [
        # Reshape(input_shape),
        Convolution2D(8, conv_size, conv_size, activation='relu', border_mode='same'),
        MaxPooling2D((conv_size-1, conv_size-1)),
        Convolution2D(16, conv_size, conv_size, activation='relu', border_mode='same'),
        MaxPooling2D((conv_size-1, conv_size-1)),
        Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same'),
        MaxPooling2D((conv_size-1, conv_size-1)),
        Flatten(),
    ]

    decode_layers = [
        Dense(32*8*8),
        Reshape((32, 8, 8),),
        UpSampling2D((conv_size-1, conv_size-1)),
        Convolution2D(16, conv_size, conv_size, activation='relu', border_mode='same'),
        UpSampling2D((conv_size-1, conv_size-1)),
        Convolution2D(8, conv_size, conv_size, activation='relu', border_mode='same'),
        UpSampling2D((conv_size-1, conv_size-1)),
        Convolution2D(color_type, conv_size, conv_size, activation='sigmoid', border_mode='same'),
    ]

elif 0:
    conv_size = 5
    conv_name = ('(32_5)(64_5)(128_5)')#TODO: generate this
    model_name = 'conv%s' % (conv_name)

    layers = [32, 64, 128]
    encode_layers = []

    for layer_size in layers:
        encode_layers.append(Convolution2D(layer_size, conv_size, conv_size, activation='relu', border_mode='same'))
        encode_layers.append(MaxPooling2D((conv_size-1, conv_size-1)))
       
    encode_layers.append(Flatten())

    smallest_rows = int(img_rows / (conv_size-1)**len(layers))
    smallest_cols = int(img_cols / (conv_size-1)**len(layers))

    decode_layers = [
        Dense(layers[-1]*smallest_rows*smallest_cols),
        Reshape((layers[-1], smallest_rows, smallest_cols),),
    ]

    for layer_size in reversed(layers[:-1]):
        decode_layers.append(UpSampling2D((conv_size-1, conv_size-1)))
        decode_layers.append(Convolution2D(layer_size, conv_size, conv_size, activation='relu', border_mode='same'))

    decode_layers.append(UpSampling2D((conv_size-1, conv_size-1)))
    decode_layers.append(Convolution2D(color_type, conv_size, conv_size, activation='sigmoid', border_mode='same'))

elif 0:
    conv_size = 3
    conv_name = ('(32_3)(64_3)(128_3)')#TODO: generate this
    model_name = 'conv%s' % (conv_name)

    layers = [32, 64, 128]
    encode_layers = []

    for layer_size in layers:
        encode_layers.append(Convolution2D(layer_size, conv_size, conv_size, activation='relu', border_mode='same'))
        encode_layers.append(MaxPooling2D((conv_size-1, conv_size-1)))
       
    encode_layers.append(Flatten())

    smallest_rows = int(img_rows / (conv_size-1)**len(layers))
    smallest_cols = int(img_cols / (conv_size-1)**len(layers))

    decode_layers = [
        Dense(layers[-1]*smallest_rows*smallest_cols),
        Reshape((layers[-1], smallest_rows, smallest_cols),),
    ]

    for layer_size in reversed(layers[:-1]):
        decode_layers.append(UpSampling2D((conv_size-1, conv_size-1)))
        decode_layers.append(Convolution2D(layer_size, conv_size, conv_size, activation='relu', border_mode='same'))

    decode_layers.append(UpSampling2D((conv_size-1, conv_size-1)))
    decode_layers.append(Convolution2D(color_type, conv_size, conv_size, activation='sigmoid', border_mode='same'))

elif 1:
    conv_size = 5
    subsample_size=2
    conv_name = ('(32_3)(64_3)(128_3)(256_3')#TODO: generate this
    model_name = 'conv%s_subsample' % (conv_name)
    if add_derivative: model_name += 'deriv'

    layers = [32, 64, 128, 256]
    encode_layers = []

    if add_derivative:
        encode_layers.append(Remove_deriv_layer)

    for layer_size in layers:
        encode_layers.append(Convolution2D(layer_size, conv_size, conv_size, 
                                activation='relu', border_mode='same', 
                                subsample=(subsample_size,subsample_size)))
        
    encode_layers.append(Flatten())

    smallest_rows = int(img_rows / (subsample_size)**len(layers))
    smallest_cols = int(img_cols / (subsample_size)**len(layers))

    decode_layers = [
        Dense(layers[-1]*smallest_rows*smallest_cols),
        Reshape((layers[-1], smallest_rows, smallest_cols),),
    ]

    for layer_size in reversed(layers[:-1]):
        decode_layers.append(UpSampling2D((subsample_size, subsample_size)))
        decode_layers.append(Convolution2D(layer_size, conv_size, conv_size, activation='relu', border_mode='same'))

    decode_layers.append(UpSampling2D((subsample_size, subsample_size)))
    decode_layers.append(Convolution2D(color_type, conv_size, conv_size, activation='sigmoid', border_mode='same'))

elif 0:
    conv_size = 5
    subsample_size=2
    conv_name = ('(32_3)(64_3)(128_3)')#TODO: generate this
    model_name = 'conv%s_subsample' % (conv_name)
    if add_derivative: model_name += 'deriv'

    layers = [32, 64, 128]
    encode_layers = []

    for layer_size in layers:
        encode_layers.append(Convolution2D(layer_size, conv_size, conv_size, 
                                activation='relu', border_mode='same', 
                                subsample=(subsample_size,subsample_size)))
        
    encode_layers.append(Flatten())

    smallest_rows = int(img_rows / (subsample_size)**len(layers))
    smallest_cols = int(img_cols / (subsample_size)**len(layers))

    decode_layers = [
        Dense(layers[-1]*smallest_rows*smallest_cols),
        Reshape((layers[-1], smallest_rows, smallest_cols),),
    ]

    for layer_size in reversed(layers[:-1]):
        decode_layers.append(UpSampling2D((subsample_size, subsample_size)))
        decode_layers.append(Convolution2D(layer_size, conv_size, conv_size, activation='relu', border_mode='same'))

    decode_layers.append(UpSampling2D((subsample_size, subsample_size)))
    decode_layers.append(Convolution2D(color_type, conv_size, conv_size, activation='sigmoid', border_mode='same'))

elif 1:
    conv_size = 3
    subsample_size=2

    dense_layers = [1000]

    conv_name = ('(32_3)(64_3)(128_3)(256_3')#TODO: generate this
    model_name = 'conv%s_subsample_dense%s' % (conv_name, str(dense_layers))

    layers = [32, 64, 128, 256]
    encode_layers = []


    for layer_size in layers:
        encode_layers.append(Convolution2D(layer_size, conv_size, conv_size,
                                activation='relu', border_mode='same',
                                subsample=(subsample_size,subsample_size)))

    encode_layers.append(Flatten())
    for dense_layer in dense_layers:
        encode_layers.append(Dense(dense_layer, activation='relu'))


    smallest_rows = int(img_rows / (subsample_size)**len(layers))
    smallest_cols = int(img_cols / (subsample_size)**len(layers))

    decode_layers = [
        Dense(dense_layers[-1], activation='relu')
    ]

    for dense_layer in dense_layers[:-1]:
        decode_layers.append(Dense(dense_layer, activation='relu'))

    decode_layers.append(Dense(layers[-1] * smallest_rows * smallest_cols, activation='relu'))

    decode_layers.append(Reshape((layers[-1], smallest_rows, smallest_cols),))
    
    for layer_size in reversed(layers[:-1]):
        decode_layers.append(UpSampling2D((subsample_size, subsample_size)))
        decode_layers.append(Convolution2D(layer_size, conv_size, conv_size, activation='relu', border_mode='same'))

    decode_layers.append(UpSampling2D((subsample_size, subsample_size)))
    decode_layers.append(Convolution2D(color_type, conv_size, conv_size, activation='sigmoid', border_mode='same'))


def remove_deriv(args):
    return args[:3]

Remove_deriv_layer = Lambda(remove_deriv, output_shape=input_shape)


def get_deriv(args):
    return args

   
c_type = 2*color_type if add_derivative else color_type
x_input = Input(batch_shape=(batch_size, c_type, img_rows, img_cols))

x = x_input
print(x)

if add_derivative:
    x = Remove_deriv_layer(x)

for encode_layer in encode_layers:
    x = encode_layer(x)

z_mean = Dense(latent_dim)(x)
z_log_std = Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])

decode_proc = z
for decode_layer in decode_layers:
    decode_proc = decode_layer(decode_proc)

def merge_func(args):
    return K.concatenate(args, axis=1)

if add_derivative and True:
    deriv_layer = Lambda(get_deriv, output_shape=(input_shape[0], input_shape[1], input_shape[2]))(decode_proc)
    print(deriv_layer)
    print(decode_proc)

    #decode_proc = Lambda(merge_func, output_shape=input_shape)([decode_proc, deriv_layer])

    decode_proc = Merge([decode_proc, deriv_layer], mode='concat', concat_axis=1)

decode_out = decode_proc

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std))#, axis=-1)
    return xent_loss + kl_loss
    
vae = Model(x_input, decode_out)
vae.summary(line_length = 150)
vae.compile(optimizer='rmsprop', loss=vae_loss)

if 0:
    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    data_name = 'mnist'
elif 1:
    x_train, y_train, train_id, driver_id, unique_drivers = run_keras_cv_drivers_v2.read_and_normalize_train_data(img_rows, img_cols, color_type)
    x_train_fit = x_train[:int(len(x_train)/ batch_size) * batch_size]
    y_train_fit = y_train[:int(len(y_train) / batch_size) * batch_size]

    data_name='sf_train_%dx%d' % (img_rows, img_cols)

model_folder = 'conv_model_%s_latent%d' % (model_name, latent_dim)
model_data_folder = '%s/%s' % (model_folder, data_name)
model_path = '%s/epoch_{epoch:03d}.h5' % (model_data_folder)
print(model_folder)
print(model_data_folder)
print(model_path)

if not os.path.exists(model_folder): os.mkdir(model_folder)
if not os.path.exists(model_data_folder): os.mkdir(model_data_folder)

start_epoch = nb_epoch - 1
while start_epoch > 0:
    epoch_path = model_path.format(epoch = start_epoch) #saved as 0 based
    if os.path.exists(epoch_path):
        vae.load_weights(epoch_path)
        start_epoch += 1 #files are 0 based, everything else is # epochs based
        break
    else:
        pass

    start_epoch -= 1

if start_epoch < nb_epoch:    
    print('nb_epoch: %d' % nb_epoch)
    print('start epoch: %d' % start_epoch)
    epochs_remaining = nb_epoch - start_epoch
    print('training for %d more iterations' % (epochs_remaining))
    callbacks = [
            ModelCheckpoint(model_path, save_best_only=False, verbose=0),
        ]


    if add_derivative:
        train_data = np.zeros((x_train_fit.shape[0], 2*x_train_fit.shape[1], x_train_fit.shape[2], x_train_fit.shape[3]), dtype='float32')
        train_data[:,:3, :, :] = x_train_fit
        train_data[:,3:, :, :] = x_train_fit
    else:
        train_data = x_train_fit

    print(train_data.shape)
    print(x_train_fit.shape)

    vae.fit(train_data, x_train_fit,
            shuffle=True,
            nb_epoch=epochs_remaining,
            batch_size=batch_size,
            verbose=True,
            callbacks = callbacks,
            epoch_offset = start_epoch
            )

    
encoder = Model(x_input, z_mean)
encoder.compile(optimizer='sgd', loss='mse')

if 1:   # build a model to project inputs on the latent space
    if 1:   #plot a random subset of data

        # display a 2D plot of the digit classes in the latent space
        encoded_data = encoder.predict(x_train, batch_size=batch_size)
        labels = categorical_to_dense(y_train)

        if latent_dim == 2:
            plt.figure(figsize=(6, 6))
            plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels)
            plt.colorbar()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            n = 2000
            skip = len(encoded_data) / n
            ax.scatter(encoded_data[::skip, 0], encoded_data[::skip, 1], zs=encoded_data[::skip, 2], c=labels[::skip])
        
        plt.title('Random drivers')
        plt.show()

    if 1: #plot for a single driver
        driver_index = 0
        X_train, Y_train, train_index = run_keras_cv_drivers_v2.copy_selected_drivers(x_train, y_train, driver_id, 
                [unique_drivers[driver_index]])
        
        encoded_data = encoder.predict(X_train, batch_size=batch_size)
        labels = categorical_to_dense(Y_train)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(encoded_data[:, 0], encoded_data[:, 1], zs=encoded_data[:, 2], c=labels[:])
        plt.title('Driver %s' % unique_drivers[driver_index])
        plt.show()


    if 1: #plot by drivers for a single class
        label_index = 0
        all_labels = categorical_to_dense(y_train)

        indices = np.where(all_labels == label_index)[0]

        input_data = x_train[indices]

        encoded_data = encoder.predict(input_data, batch_size=batch_size)

        le = preprocessing.LabelEncoder()
        labels = le.fit_transform([driver_id[i] for i in indices])

        # indices = np.where(labels < 10)[0]
        # labels = labels[indices]
        # encoded_data = encoded_data[indices]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(encoded_data[:, 0], encoded_data[:, 1], zs=encoded_data[:, 2], c=labels[:])
        plt.title('Class label %s' % label_index)
        plt.show()

if 0: #add a softmax layer on top of encoder
    softmax = Dense(num_classes, activation='softmax')(z_mean)

    classifier = Model(x_input, softmax)
    classifier.summary()


    optimizer = Adam(lr=1e-3)
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = []

    classifier.fit(x_train, y_train,
            shuffle=True,
            nb_epoch = 20,
            batch_size = batch_size,
            validation_data = (x_train, y_train),
            verbose=True,
            callbacks = callbacks,
            )

if 0:   #add a softmax, validatation split by drivers
    x = z_mean
    x = Dropout(0.5)(x)
    
    x = Dense(num_classes, activation='softmax')(x)
	
    # optimizer = Adam(lr=1e-3)
    optimizer = SGD(lr=1e-4, momentum = 0.9, nesterov=True)
    classifier = Model(x_input, x)
    classifier.summary()
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    kf = KFold(len(unique_drivers), n_folds=4, shuffle=True, random_state=51)
    for num_fold, (train_drivers, test_drivers) in enumerate(kf):

        vae.load_weights(epoch_path)

        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = run_keras_cv_drivers_v2.copy_selected_drivers(x_train, y_train, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = run_keras_cv_drivers_v2.copy_selected_drivers(x_train, y_train, driver_id, unique_list_valid)


        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'))

        classifier.fit(X_train, Y_train,
                shuffle=True,
                nb_epoch = 200,
                batch_size = batch_size,
                validation_data = (X_valid, Y_valid),
                verbose=True,
                callbacks = callbacks,
                )

if 1:   #encode data, train a simple model on top of it
    
    encoded_data = encoder.predict(x_train)

    kf = KFold(len(unique_drivers), n_folds=4, shuffle=True, random_state=51)
    for num_fold, (train_drivers, test_drivers) in enumerate(kf):

        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = run_keras_cv_drivers_v2.copy_selected_drivers(encoded_data, y_train, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = run_keras_cv_drivers_v2.copy_selected_drivers(encoded_data, y_train, driver_id, unique_list_valid)


        encode_input = Input(encoded_data.shape[1:])

        x = encode_input
        x = Dropout(0.5)(x)
        #x = GaussianNoise(epsilon_std)(x)
        #x = Dense(100, activation='relu')(x)
        #x = Dropout(0.5)(x)

        #x = Dense(1000, activation='relu')(x)
        #x = Dropout(0.5)(x)

        #x = Dense(100, activation='relu')(x)
        #x = Dropout(0.5)(x)

        x = Dense(num_classes, activation='softmax')(x)

        # optimizer = Adam(lr=1e-3)
        # optimizer = 'rmsprop'
        optimizer = SGD(lr=2e-2)
        classifier = Model(encode_input, x)
        classifier.summary()
        classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = []
        #callbacks.append(EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'))
        
        classifier.fit(X_train, Y_train,
                shuffle=True,
                nb_epoch = 200,
                batch_size = batch_size,
                validation_data = (X_valid, Y_valid),
                verbose=True,
                callbacks = callbacks,
                )



# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))

decoder_proc = decoder_input

for decode_layer in decode_layers:
    decoder_proc = decode_layer(decoder_proc)

generator = Model(decoder_input, decoder_proc)
#generator.summary()
generator.compile(optimizer='sgd', loss='mse')



if 0: #plot some random spots from the latent space
    while 1:

        # display a 2D manifold of the digits
        n = 5  # figure with 15x15 digits
        figure = np.zeros((img_rows * n, img_cols * n))
        # we will sample n points within [-15, 15] standard deviations
        grid_x = np.linspace(-15, 15, n)
        grid_y = np.linspace(-15, 15, n)

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.random.normal(0, epsilon_std, (1, latent_dim))
                z_sample[0,0] = xi * epsilon_std
                z_sample[0,1] = yi * epsilon_std

                x_decoded = generator.predict(z_sample)
                digit = x_decoded[0].reshape(3, img_rows, img_cols)

                plt.imshow(np.transpose(digit, (1, 2, 0)))
                plt.show()

if 0: #build an image with scrollbars to adjust latent space
    
    window_name = 'Decoded output'
    latent_vars = np.zeros((1, latent_dim))

    trackbar_range = 1000

    def update_output_img():
        decoded = generator.predict(latent_vars)
        x_decoded = np.transpose(decoded[0], (1, 2, 0))
        cv2.imshow(window_name, x_decoded)

    range_num_std = 5
    def trackbar_callback_base(dim, value):
        float_value = range_num_std * epsilon_std * (2 * (value - trackbar_range/2) / trackbar_range)
        latent_vars[0][dim] = float_value
        update_output_img() 

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for i in range(min(10, latent_dim)):
        cv2.createTrackbar(str(i), window_name, 
            int(trackbar_range / 2), trackbar_range,
            functools.partial(trackbar_callback_base, i)
            )

    update_output_img()
    cv2.waitKey(0)

if 0: #do some encoding + decoding and plot
    while 1:
        input_img = x_train[np.random.uniform(len(x_train))]
        full_input_img = np.zeros((batch_size, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
        full_input_img[0] = input_img
        decoded_img = vae.predict(full_input_img, batch_size = batch_size)[0]

        plt.subplot(2,1,1)
        plt.imshow(np.transpose(input_img, (1, 2, 0)))
        plt.subplot(2, 1, 2)
        plt.imshow(np.transpose(decoded_img, (1, 2, 0)))
        plt.show()
