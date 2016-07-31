from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dropout, Activation, Reshape, Merge
from keras.utils import np_utils
from keras.optimizers import *

import vgg16_efficiency

def build_vae_models(encode_layers, decode_layers, img_rows, img_cols, color_type, latent_dim, batch_size, epsilon_std):

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

	optimizer = RMSprop(lr=1e-4)
	# optimizer = 'rmsprop'

	vae.compile(optimizer=optimizer, loss=vae_loss)



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

def set_vgg16_weights(vae, encoder, decoder):
	vgg16_efficiency.set_vgg16_model_2_weights(encoder, set_last_layer=False)

	# for e, d in zip(encoder.layers[1:], reversed(decoder.layers)):
	# 	print(e, d)
	# 	print(e.get_weights()[0].T.shape)
	# 	if len(e.get_weights()) > 0:
	# 	    print('setting decoder weights. shape: %s' % str(e.get_weights()[0].shape))
	#
	# 	d.set_weights(e.get_weights())

def set_vgg16_encoder_weights(encoder_layers):
    f = h5py.File('vgg16_weights.h5')
    model_k = 0
    model_num_layers = len(encoder_layers)

    for k in range(f.attrs['nb_layers']):
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        if len(weights) > 0:
            while model_k < model_num_layers and len(encoder_layers[model_k].get_weights()) == 0:
                model_k += 1
                print('skipping model layer %d' % model_k)

            #print('setting weights from full model layer %d to layer %d' % (k, model_k))
            if model_k == model_num_layers: break
            print('setting layer %d / %d' % (model_k, model_num_layers))
            encoder_layers[model_k].set_weights(weights)
            model_k += 1
    f.close()
