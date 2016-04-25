
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from collections import OrderedDict
import copy
from six.moves import zip

import inspect

from keras.layers.core import Layer


class AutoEncoder(Layer):
    '''A customizable autoencoder model.
    # Input shape
        Same as encoder input.
    # Output shape
        If `output_reconstruction = True` then dim(input) = dim(output)
        else dim(output) = dim(hidden).
    # Arguments
        encoder: A [layer](./) or [layer container](./containers.md).
        decoder: A [layer](./) or [layer container](./containers.md).
        output_reconstruction: If this is `False`,
            the output of the autoencoder is the output of
            the deepest hidden layer.
            Otherwise, the output of the final decoder layer is returned.
        weights: list of numpy arrays to set as initial weights.
    # Examples
    ```python
        from keras.layers import containers, AutoEncoder, Dense
        from keras import models
        # input shape: (nb_samples, 32)
        encoder = containers.Sequential([Dense(16, input_dim=32), Dense(8)])
        decoder = containers.Sequential([Dense(16, input_dim=8), Dense(32)])
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
        model = models.Sequential()
        model.add(autoencoder)
        # training the autoencoder:
        model.compile(optimizer='sgd', loss='mse')
        model.fit(X_train, X_train, nb_epoch=10)
        # predicting compressed representations of inputs:
        autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
        model.compile(optimizer='sgd', loss='mse')
        representations = model.predict(X_test)
        # the model is still trainable, although it now expects compressed representations as targets:
        model.fit(X_test, representations, nb_epoch=1)  # in this case the loss will be 0, so it's useless
        # to keep training against the original inputs, just switch back output_reconstruction to True:
        autoencoder.output_reconstruction = True
        model.compile(optimizer='sgd', loss='mse')
        model.fit(X_train, X_train, nb_epoch=10)
    ```
    '''
    def __init__(self, encoder, decoder, output_reconstruction=True,
                 weights=None, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

        self._output_reconstruction = output_reconstruction
        self.encoder = encoder
        self.encoder.layer_cache = self.layer_cache
        self.decoder = decoder
        self.decoder.layer_cache = self.layer_cache

        if output_reconstruction:
            self.decoder.set_previous(self.encoder)

        if weights is not None:
            self.set_weights(weights)

        super(AutoEncoder, self).__init__(**kwargs)
        self.build()

    @property
    def output_reconstruction(self):
        return self._output_reconstruction

    @output_reconstruction.setter
    def output_reconstruction(self, value):
        self._output_reconstruction = value
        self.build()

    def build(self):
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        if self.output_reconstruction:
            layers = [self.encoder, self.decoder]
        else:
            layers = [self.encoder]
        for layer in layers:
            params, regularizers, constraints, updates = layer.get_params()
            self.regularizers += regularizers
            self.updates += updates
            for p, c in zip(params, constraints):
                if p not in self.trainable_weights:
                    self.trainable_weights.append(p)
                    self.constraints.append(c)

    @property
    def layer_cache(self):
        return super(AutoEncoder, self).layer_cache

    @layer_cache.setter
    def layer_cache(self, value):
        self._layer_cache = value
        self.encoder.layer_cache = self._layer_cache
        self.decoder.layer_cache = self._layer_cache

    @property
    def shape_cache(self):
        return super(AutoEncoder, self).shape_cache

    @shape_cache.setter
    def shape_cache(self, value):
        self._shape_cache = value
        self.encoder.shape_cache = self._shape_cache
        self.decoder.shape_cache = self._shape_cache

    def set_previous(self, node, reset_weights=True):
        self.encoder.set_previous(node, reset_weights)
        if reset_weights:
            self.build()

    def get_weights(self):
        weights = []
        for layer in [self.encoder, self.decoder]:
            weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        nb_param = len(self.encoder.trainable_weights)
        self.encoder.set_weights(weights[:nb_param])
        self.decoder.set_weights(weights[nb_param:])

    def get_input(self, train=False):
        return self.encoder.get_input(train)

    @property
    def input(self):
        return self.encoder.input

    @property
    def input_shape(self):
        return self.encoder.input_shape

    @property
    def output_shape(self):
        if self.output_reconstruction:
            return self.decoder.output_shape
        else:
            return self.encoder.output_shape

    def get_output(self, train=False):
        if self.output_reconstruction:
            return self.decoder.get_output(train)
        else:
            return self.encoder.get_output(train)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'encoder_config': self.encoder.get_config(),
                'decoder_config': self.decoder.get_config(),
                'output_reconstruction': self.output_reconstruction}
