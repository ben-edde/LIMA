import tensorflow as tf
from keras.layers import (GRU, LSTM, Bidirectional, Conv1D, Conv2D, Conv3D,
                          ConvLSTM2D, Dense, Dropout, Flatten, Input,
                          MaxPooling2D, Reshape, SimpleRNN)
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


class NonLinearModelFactory:
    @classmethod
    def get_model(self, num_feature, num_label, model_name="BiGRU"):
        if model_name == "BiGRU":
            return NonLinearModelFactory.BiGRU(num_feature, num_label)

    @classmethod
    def BiGRU(self, num_feature, num_label):
        inputs = Input(shape=(num_feature, ))
        model = Reshape((num_feature, 1))(inputs)
        model = Bidirectional(GRU(50, dropout=0.2,
                                  return_sequences=True))(model)
        model = Dropout(0.4)(model)
        model = Bidirectional(GRU(50, dropout=0.2,
                                  return_sequences=True))(model)
        model = Dropout(0.4)(model)
        model = Bidirectional(GRU(50, dropout=0.2,
                                  return_sequences=False))(model)
        model = Dropout(0.4)(model)
        model = Flatten()(model)
        model = Dense(num_label)(model)
        model = Model(inputs=inputs, outputs=model)
        model.compile(loss='log_cosh', optimizer=Adam(0.0002))
        return model

    @classmethod
    def BiLSTM(self, num_feature, num_label):
        inputs = Input(shape=(num_feature, ))
        model = Reshape((num_feature, 1))(inputs)
        model = Bidirectional(LSTM(50, dropout=0.2,
                                   return_sequences=True))(model)
        model = Dropout(0.4)(model)
        model = Bidirectional(LSTM(50, dropout=0.2,
                                   return_sequences=True))(model)
        model = Dropout(0.4)(model)
        model = Bidirectional(LSTM(50, dropout=0.2,
                                   return_sequences=False))(model)
        model = Dropout(0.4)(model)
        model = Flatten()(model)
        model = Dense(num_label)(model)
        model = Model(inputs=inputs, outputs=model)
        model.compile(loss='log_cosh', optimizer=Adam(0.0002))
        return model

    @classmethod
    def BiRNN(self, num_feature, num_label):
        inputs = Input(shape=(num_feature, ))
        model = Reshape((num_feature, 1))(inputs)
        model = Bidirectional(SimpleRNN(50, dropout=0.2,
                                        return_sequences=True))(model)
        model = Dropout(0.4)(model)
        model = Bidirectional(SimpleRNN(50, dropout=0.2,
                                        return_sequences=True))(model)
        model = Dropout(0.4)(model)
        model = Bidirectional(
            SimpleRNN(50, dropout=0.2, return_sequences=False))(model)
        model = Dropout(0.4)(model)
        model = Flatten()(model)
        model = Dense(num_label)(model)
        model = Model(inputs=inputs, outputs=model)
        model.compile(loss='log_cosh', optimizer=Adam(0.0002))
        return model
