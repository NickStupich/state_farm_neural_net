'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers import Flatten, Dropout, Activation, Reshape
from keras.utils import np_utils

class VAEContainer(object):
    def __init__(self, 
                encoder_layer, 
                decoder_layers,
                name
                ):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.name = name


batch_size = 16
original_dim = 784
latent_dim = 3
epsilon_std = 0.01
nb_epoch = 3000
num_classes = 10

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

    # vae_specs = VAEContainer(encode_layers, decode_layers, model_name)

elif 1:
    conv_size = 3
    conv_name = ('(8_3)(16_3)')
    model_name = 'conv%s' % (conv_name)

    encode_layers = [
        Reshape((1, 28, 28)),
        Convolution2D(8, conv_size, conv_size, activation='relu', border_mode='same'),
        MaxPooling2D((conv_size-1, conv_size-1)),
        Convolution2D(16, conv_size, conv_size, activation='relu', border_mode='same'),
        MaxPooling2D((conv_size-1, conv_size-1)),
        Flatten(),
    ]

    decode_layers = [
        Dense(16*7*7, activation='sigmoid'),
        Reshape((16, 7, 7)),
        UpSampling2D((conv_size-1, conv_size-1)),
        Convolution2D(8, conv_size, conv_size, activation='sigmoid', border_mode='same'),
        UpSampling2D((conv_size-1, conv_size-1)),
        Convolution2D(1, conv_size, conv_size, activation='sigmoid', border_mode='same'),
        Flatten(),
    ]

    # vae_specs = VAEContainer(encode_layers, decode_layers, model_name)


x_input = Input(batch_shape=(batch_size, original_dim))

x = x_input
for encode_layer in encode_layers:
    x = encode_layer(x)

z_mean = Dense(latent_dim)(x)
z_log_std = Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_std])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])

decode_proc = z
for decode_layer in decode_layers:
    decode_proc = decode_layer(decode_proc)

decode_out = decode_proc

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std))#, axis=-1)
    print(xent_loss)
    print(kl_loss)
    print(xent_loss.shape)
    print(kl_loss.shape)
    return xent_loss + kl_loss

vae = Model(x_input, decode_out)
vae.summary(line_length = 150)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


model_path = 'conv_model_%s_latent%d/mnist_{epoch:03d}.h5' % (model_name, latent_dim)
model_path_best = 'conv_model_%s_latent%d/mnist_best.h5' % (model_name, latent_dim)
print('model path: %s' % model_path)

if 0:

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
                ModelCheckpoint(model_path, save_best_only=False, monitor='val_loss', verbose=0),
                ModelCheckpoint(model_path_best, save_best_only=True, monitor='val_loss', verbose=0),
                EarlyStopping(monitor='val_loss', patience=1000, verbose=0)
            ]

        vae.fit(x_train, x_train,
                shuffle=True,
                nb_epoch=epochs_remaining,
                batch_size=batch_size,
                validation_data=(x_test, x_test),
                verbose=True,
                callbacks = callbacks,
                epoch_offset = start_epoch
                )

vae.load_weights(model_path_best)

if 1:
    # build a model to project inputs on the latent space
    encoder = Model(x_input, z_mean)
    encoder.compile(optimizer='sgd', loss='mse')

    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

    if latent_dim == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
        plt.colorbar()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], zs=x_test_encoded[:, 2], c=y_test)

    plt.show()

if 1: #add a softmax layer on top of encoder
    softmax = Dense(num_classes, activation='softmax')(z_mean)

    classifier = Model(x_input, softmax)
    classifier.summary()
    classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = []

    classifier.fit(x_train, np_utils.to_categorical(y_train, num_classes),
            shuffle=True,
            nb_epoch = 20,
            batch_size = batch_size,
            validation_data = (x_test, np_utils.to_categorical(y_test, num_classes)),
            verbose=True,
            callbacks = callbacks,
            )


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))

decoder_proc = decoder_input

for decode_layer in decode_layers:
    decoder_proc = decode_layer(decoder_proc)

generator = Model(decoder_input, decoder_proc)
generator.summary()
generator.compile(optimizer='sgd', loss='mse')


if latent_dim == 2:
    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # we will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * epsilon_std
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()