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


def load_custom_model(path_to_model_json, path_to_model_h5):
    json_file = open(path_to_model_json, 'r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)
    model.load_weights(path_to_model_h5)
    return model


def get_bagged_models(verbose):
    models = []
    for i in range(1, 6):
        models.append(load_custom_model("./models/next_event_model_%d.json" % i,
                                        "./models/next_event_model_%d.h5" % i))
        if verbose:
            print('Loaded model next_event %d/%d' % (i, 5))
    return models


class NextMoveNet():
    def __init__(self, verbose=1):
        self.models = get_bagged_models(verbose=verbose)
        pass

    def predict_proba(self, input):

        switch, pos = self.models[0].predict(input)


        for i in range(1, 5):
            s, p = self.models[i].predict(input)
            switch += s
            pos += p

        return switch / 5, pos / 5

    def predict(self, input):

        switch, pos = self.predict_proba(input)
        return int(switch>.5), pos