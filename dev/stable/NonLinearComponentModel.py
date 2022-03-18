import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (GRU, LSTM, Bidirectional, Conv1D, Conv2D, Conv3D,
                          ConvLSTM2D, Dense, Dropout, Flatten, Input,
                          MaxPooling2D, Reshape, SimpleRNN)
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from matplotlib import pyplot
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

# set random seed
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class NonLinearComponentModel:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.model = None
        self.preprocessing()

    def stationary_test(self, data):
        for i in range(data.shape[1]):
            test_result = adfuller(data[:, i])
            if test_result[1] > 0.05:
                return False
        return True

    def stl_decompose(self, data):
        stl = STL(data, 10)
        res = stl.fit()
        non_linear_trend = res.trend.reshape(-1, 1)
        non_linear_season = res.seasonal.reshape(-1, 1)
        non_linear_residual = res.resid.reshape(-1, 1)
        non_linear_y = np.concatenate(
            [non_linear_trend, non_linear_season, non_linear_residual], axis=1)
        return non_linear_y

    def differencing(self, data):
        diffed_data = np.diff(data, axis=0)
        return diffed_data

    def ensure_stationary(self, data):
        diff_order = 0
        feature = data.feature
        label = data.label
        while not self.stationary_test(feature):
            feature = self.differencing(feature)
            diff_order += 1
        label = label[diff_order:]
        return feature, label, diff_order

    def set_model(self, model_name):
        self.model = self.compose_model(self.dataset.train_X.shape[-1],
                                        self.dataset.train_y.shape[-1])

    def train(self):
        self.prepare_env()
        history = self.model.fit(self.dataset.train_X,
                                 self.dataset.train_y,
                                 epochs=200,
                                 batch_size=40,
                                 verbose=1,
                                 shuffle=False)
        # pyplot.plot(history.history['loss'], label='train')
        # pyplot.legend()
        # pyplot.show()

    def predict(self, feature):
        return self.model.predict(feature)

    def prepare_env(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)

    def compose_model(self, num_feature, num_label):
        ts_inputs = Input(shape=(num_feature, ))
        ts_model = Reshape((num_feature, 1))(ts_inputs)
        ts_model = Bidirectional(GRU(50, dropout=0.2,
                                     return_sequences=True))(ts_model)
        ts_model = Dropout(0.4)(ts_model)
        ts_model = Bidirectional(GRU(50, dropout=0.2,
                                     return_sequences=True))(ts_model)
        ts_model = Dropout(0.4)(ts_model)
        ts_model = Bidirectional(GRU(50, dropout=0.2,
                                     return_sequences=False))(ts_model)
        ts_model = Dropout(0.4)(ts_model)
        ts_model = Flatten()(ts_model)
        ts_model = Dense(num_label)(ts_model)
        ts_model = Model(inputs=ts_inputs, outputs=ts_model)
        ts_model.compile(loss='log_cosh', optimizer=Adam(0.0002))
        return ts_model

    def preprocessing(self):
        self.dataset.label = self.stl_decompose(self.dataset.label)
        self.dataset.feature, self.dataset.label, self.diff_order = self.ensure_stationary(
            self.dataset)
        self.dataset.update()
