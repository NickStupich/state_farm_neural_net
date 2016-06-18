'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers import Flatten, Dropout, Activation, Reshape


batch_size = 16
original_dim = 784
latent_dim = 2
intermediate_dim = 128
epsilon_std = 0.01
nb_epoch = 1000

conv_size = 5

x = Input(batch_shape=(batch_size, original_dim))
x2 = Reshape((1, 28, 28))(x)
x2 = Convolution2D(8, conv_size, conv_size, activation='relu', border_mode='same')(x2)
x2 = MaxPooling2D((conv_size-1, conv_size-1))(x2)
x2 = Flatten()(x2)
h = Dense(intermediate_dim, activation='relu')(x2)
z_mean = Dense(latent_dim)(h)
z_log_std = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_std])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])

# we instantiate these layers separately so as to reuse them later

decode_layers = [
    Dense(intermediate_dim, activation='relu'),
    Dense((int(8*28*28 / ((conv_size-1)**2))), activation='sigmoid'),
    Reshape((8, int(28 / (conv_size-1)), int(28 / (conv_size-1)))),
    UpSampling2D((conv_size-1, conv_size-1)),
    Convolution2D(1, conv_size, conv_size, activation='sigmoid', border_mode='same'),
    Flatten()
    ]

# decode_layers =

decode_proc = z
for decode_layer in decode_layers:
    decode_proc = decode_layer(decode_proc)

decode_out = decode_proc

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, decode_out)
vae.summary()
vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


model_path = 'conv_models/mnist_{epoch:03d}.h5' 

start_epoch = nb_epoch - 1
while start_epoch > 0:
    epoch_path = model_path.format(epoch = start_epoch) #saved as 0 based
    if os.path.exists(epoch_path):
        print('found')
        vae.load_weights(epoch_path)
        start_epoch += 1 #files are 0 based, everything else is # epochs based
        break
    else:
        print('not found')
        pass

    start_epoch -= 1


if start_epoch < nb_epoch:    
    print('nb_epoch: %d' % nb_epoch)
    print('start epoch: %d' % start_epoch)
    epochs_remaining = nb_epoch - start_epoch
    print('training for %d more iterations' % (epochs_remaining))
    callbacks = [
            ModelCheckpoint(model_path, save_best_only=False, monitor='val_loss', verbose=0),
            EarlyStopping(monitor='val_loss', patience=10, verbose=0)
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

if 1:
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    encoder.compile(optimizer='sgd', loss='mse')

    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))

decoder_proc = decoder_input

for decode_layer in decode_layers:
    decoder_proc = decode_layer(decoder_proc)

generator = Model(decoder_input, decoder_proc)
generator.summary()
generator.compile(optimizer='sgd', loss='mse')

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