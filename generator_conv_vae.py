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
import vae_helpers
import use_vae_experiments

batch_size = 64
latent_dim = 3
epsilon_std = 0.01
nb_epoch = 1
num_classes = 10

input_shape = (3, 128, 128)
color_type, img_rows, img_cols = input_shape

# encode_layers, decode_layers, model_name = conv_vae_models.get_conv_vae_model(layers = [32, 64, 128, 256],
# 			conv_size=5, subsample=2, model_name='test1', img_rows = img_rows, img_cols = img_cols, color_type=color_type)

encode_layers, decode_layers, model_name = conv_vae_models.get_vgg16_conv_model(img_rows = img_rows, img_cols = img_cols, color_type=color_type)

vae, encoder, decoder = vae_helpers.build_vae_models(encode_layers, decode_layers, img_rows, img_cols, color_type, latent_dim, batch_size, epsilon_std)

if 1:
	start = datetime.datetime.now()
	all_data = train_data_generator.get_unlabelled_data(img_rows, img_cols, color_type)
	end = datetime.datetime.now()
	print('data loading time: %s' % (end - start))
	print('array size: %s MB' % str(all_data.nbytes / (2**20)))
else:
	print("***WARNING: FAKE DATA***")
	all_data = np.zeros((16*450, color_type, img_rows, img_cols))

#all_data = all_data[:79650] #to make batch sizes work out
#all_data = all_data[:22424 - (22424 % batch_size)]	#train only

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

if 0:
	n = 79726 - 79726 % batch_size
	print('test cost: %s' % str(vae.evaluate(all_data[:n], all_data[:n], batch_size=batch_size)))

	n = (len(all_data) - 79726) - (len(all_data) - 79726) % batch_size
	print('train cost: %s' % str(vae.evaluate(all_data[79726:79726+n], all_data[79726:79726+n], batch_size=batch_size)))

# use_vae_experiments.cluster_and_classify(encoder, all_data, img_rows, img_cols, color_type)
use_vae_experiments.plot_top_clusters(encoder, all_data, img_rows, img_cols, color_type)
