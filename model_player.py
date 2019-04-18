import json
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import (Bidirectional, GRU, Layer, Input, Embedding,
                          Dropout, Dense, TimeDistributed, concatenate,
                          BatchNormalization, Reshape, Flatten, add,
                          subtract, GlobalAveragePooling1D)

from keras import initializers, regularizers, constraints
from keras.models import Model, model_from_json
from keras.backend.tensorflow_backend import _to_tensor


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]

    def get_config(self):
        config = {"return_coefficients": self.return_coefficients,
                  "W_regularizer": self.W_regularizer,
                  "u_regularizer": self.u_regularizer,
                  "b_regularizer": self.b_regularizer,
                  "W_constraint": self.W_constraint,
                  "u_constraint": self.u_constraint,
                  "b_constraint": self.b_constraint,
                  "bias": self.bias}
        base_config = super(AttentionWithContext, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SplitLayer(Layer):

    def __init__(self, range_, **kwargs):
        self.range_ = range_
        self.start = self.range_[0]
        self.end = self.range_[1]
        self.output_dim = self.end - self.start
        super(SplitLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Be sure to call this at the end
        super(SplitLayer, self).build(input_shape)

    def call(self, x):
        return x[:, self.start:self.end]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'range_': self.range_}
        base_config = super(SplitLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_custom_model(path_to_model_json, path_to_model_h5):
    json_file = open(path_to_model_json, 'r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json,
                            custom_objects={"SplitLayer": SplitLayer,
                                            "AttentionWithContext": AttentionWithContext})
    model.load_weights(path_to_model_h5)
    return model


def get_bagged_models(verbose):
    models = []
    for i in range(1, 6):
        models.append(load_custom_model("./models/player_model_%d.json" % i,
                                        "./models/player_model_%d.h5" % i))
        if verbose:
            print('Loaded model player %d/%d' % (i, 5))
    return models


class WhosThatNet():
    def __init__(self, verbose=1):
        self.models = get_bagged_models(verbose=verbose)
        pass

    def predict_proba(self, input):

        r = self.models[0].predict(input)[-1]

        for i in range(1, 5):
            r = self.models[i].predict(input)[-1]

        return r / 5

    def predict(self, input):

        return np.argmax(self.predict_proba(input).squeeze())