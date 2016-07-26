from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dropout, Activation, Reshape, Merge
from keras.utils import np_utils
from keras.optimizers import *

def build_vae_models(encode_layers, decode_layers, img_rows, img_cols, color_type):

	x_input = Input(shape=(color_type, img_rows, img_cols))
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

	def vae_loss(x, x_decoded_mean):
	    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
	    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std))#, axis=-1)
	    return xent_loss + kl_loss
	    

	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])

	decode_x = z
	for decode_layer in decode_layers:
	    decode_x = decode_layer(decode_x)

	vae = Model(x_input, decode_x)
	vae.summary()
	vae.compile(optimizer='rmsprop', loss=vae_loss)



	encoder = Model(x_input, z_mean)
	encoder.compile(optimizer='sgd', loss='mse')

	decoder_input = Input(shape=(latent_dim,))
	decoder_proc = decoder_input
	for decode_layer in decode_layers:
		decoder_proc = decode_layer(decoder_proc)


	decoder = Model(decoder_input, decoder_proc)
	decoder.compile(optimizer='sgd', loss='mse')


	print('all done, got vae, encoder, decoder')


	return vae, encoder, decoder