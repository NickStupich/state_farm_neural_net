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
from keras.layers import Flatten, Dropout, Activation, Reshape, Merge
from keras.utils import np_utils
from keras.optimizers import *
from keras.layers.noise import GaussianNoise

from sklearn.cluster import KMeans

import datetime
import conv_vae_models

import train_data_generator
import run_keras_cv_drivers_v2

batch_size = 450
latent_dim = 3
epsilon_std = 0.01
nb_epoch = 48
num_classes = 10

input_shape = (3, 128, 128)
color_type, img_rows, img_cols = input_shape

if 1:
	encode_layers, decode_layers, model_name = conv_vae_models.get_conv_vae_model(layers = [32, 64, 128, 256], 
			conv_size=5, subsample=2, model_name='test1', img_rows = img_rows, img_cols = img_cols, color_type=color_type)


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

start = datetime.datetime.now()
all_data = train_data_generator.get_unlabelled_data(img_rows, img_cols, color_type)
end = datetime.datetime.now()
print('data loading time: %s' % (end - start))
print('array size: %s MB' % str(all_data.nbytes / (2**20)))


model_folder = 'vae_gen_%s_latent%d' % (model_name, latent_dim)
model_path = '%s/epoch_{epoch:03d}.h5' % (model_folder)
print(model_folder)
print(model_path)

if not os.path.exists(model_folder): os.mkdir(model_folder)

start_epoch = nb_epoch - 1
while start_epoch >= 0:
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

    vae.fit(all_data, all_data,
            shuffle=True,
            nb_epoch=epochs_remaining,
            batch_size=batch_size,
            verbose=True,
            callbacks = callbacks
            )
    
encoder = Model(x_input, z_mean)
encoder.compile(optimizer='sgd', loss='mse')

decoder_input = Input(shape=(latent_dim,))
decoder_proc = decoder_input
for decode_layer in decode_layers:
    decoder_proc = decode_layer(decoder_proc)


decoder = Model(decoder_input, decoder_proc)
decoder.compile(optimizer='sgd', loss='mse')


print('all done, got vae, encoder, decoder')


if 1:   # build a model to project inputs on the latent space
	if 1:   #plot with driver ids as label

		x_train, y_train, train_id, driver_id, unique_drivers = run_keras_cv_drivers_v2.read_and_normalize_train_data(img_rows, img_cols, color_type)
		driver_id_int = list(map(lambda x: int(x.strip('p')), driver_id))
		total_driver_counts = np.zeros(100, dtype='int')
		for i in driver_id_int: total_driver_counts[i] += 1
		to_encode = x_train
		#to_encode = all_data


		# display a 2D plot of the digit classes in the latent space
		encoded_data = encoder.predict(to_encode, batch_size=batch_size)
		#labels = np.array(list(map(lambda id_str: int(id_str.strip('p')), driver_id)))
		#class_labels = np.argmax(y_train, axis=1)
		num_clusters = 10
		kmeans = KMeans(n_clusters = num_clusters)
		pred_labels = kmeans.fit_predict(encoded_data)

		print(pred_labels[:100])

		if 0:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			n = 5000
			skip = len(encoded_data) / n
			#ax.scatter(encoded_data[::skip, 0], encoded_data[::skip, 1], zs=encoded_data[::skip, 2], c=labels[::skip])
			ax.scatter(encoded_data[::skip, 0], encoded_data[::skip, 1], zs=encoded_data[::skip, 2], c=pred_labels[::skip])

			plt.title('Random drivers - colors are driver ids')
			plt.show()

		for label in range(num_clusters):
			#build a mini-autoencoder here
			indices = np.where(pred_labels == label)[0]
			
			y_train_dense = np.argmax(y_train, axis=1)
			driver_counts = np.zeros(100, dtype='int')
			class_counts = np.zeros(10, dtype='int')
			for ind in indices:
				class_counts[y_train_dense[ind]] += 1
				driver_counts[driver_id_int[ind]] += 1

			print('num indices: %d' % len(indices))
			#print(len(labels_dense))
			print('num train indices: %d' % len(np.where(indices >= 79726)[0]))
			
			unique_drivers = list(map(lambda y: y[0], filter(lambda x: x[1] > 0, enumerate(driver_counts))))
			
			print('driver_counts:')
			print('\n'.join(map(lambda x: str(x[0]) + ' - ' + str(x[1][0]) + ' / ' + str(x[1][1]), filter(lambda y: y[1][0] > 0, enumerate(zip(driver_counts, total_driver_counts))))))
			print('class counts: %s' % str(class_counts))
			print('unique drivers: %s' % str(unique_drivers))

			#fig = plt.figure()
			#ax = fig.add_subplot(111, projection='3d')

			if 0:
				ax.scatter(encoded_data[indices,0], encoded_data[indices, 1], zs=encoded_data[indices, 2])#, c=class_labels[indices])
			elif 0:
				label_x_train = np.reshape(to_encode[indices], (len(indices[0]), -1))
				print(label_x_train.shape, to_encode.shape)
				tsne = TSNE(n_components=3)
				tsne_data = tsne.fit_transform(label_x_train)
				ax.scatter(tsne_data[indices,0], tsne_data[indices, 1], zs=tsne_data[indices, 2])#, c=class_labels[indices])

			#plt.legend()
			#plt.title('Clustered driver %d - colors are class labels' % label)
			#plt.show()
