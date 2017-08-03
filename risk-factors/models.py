from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# Lasagne
from lasagne.nonlinearities import softmax

# Keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import Concatenate
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


class MultitaskEmbedding(BaseEstimator, ClassifierMixin):
    def __init__(self, depth=4, width=20, alpha=0.1,
                 embedding='raw', bypass=False):
        self.depth = depth
        self.width = width
        self.alpha = alpha
        self.embedding = embedding
        self.bypass = bypass

    def fit(self, X, y):
        train_X, val_X, train_y, val_y = train_test_split(X, y, stratify=y,
                                                          test_size=0.2,
                                                          random_state=42)

        np.random.seed(42)

        self.network = self.build_network(X, y)
        ae_trX = train_X
        ae_valX = val_X

        if self.embedding == 'raw':
            ae_trX = train_X
            ae_valX = val_X
        elif self.embedding == 'symmetric':
            ae_trX = -train_X * (2 * train_y[:, np.newaxis] - 1)
            ae_valX = -val_X * (2 * val_y[:, np.newaxis] - 1)
        elif self.embedding == 'zero':
            leaky = 0.05
            ae_trX = train_X * (leaky + (1 - leaky) * train_y)[:, np.newaxis]
            ae_valX = val_X * (leaky + (1 - leaky) * val_y)[:, np.newaxis]

        self.network.fit(train_X, [ae_trX,
                                   to_categorical(train_y, num_classes=2)],

                         epochs=500,
                         validation_data=(val_X,
                                          [ae_valX,
                                           to_categorical(val_y,
                                                          num_classes=2)]),
                         callbacks=[EarlyStopping(monitor='val_loss',
                                                  patience=100)],
                         verbose=0,
                         )

        return self

    def build_network(self, X, y):
        shared_input_layer = Input(shape=(X.shape[1],), name='input')

        shared_dropout_layer = Dropout(rate=1. / X.shape[1],
                                       name='dropout')(shared_input_layer)

        shared_last_layer = shared_dropout_layer

        for d in range(self.depth):
            shared_last_layer = Dense(self.width, activation='linear',
                                      )(shared_last_layer)
            if d + 1 == self.depth:
                shared_last_layer = PReLU(name='hidden-layer')(
                    shared_last_layer)
            else:
                shared_last_layer = PReLU()(shared_last_layer)

        # Autoencoder
        ae_last_layer = shared_last_layer
        for d in range(self.depth):
            ae_last_layer = Dense(self.width,
                                  activation='linear')(ae_last_layer)
            ae_last_layer = PReLU()(ae_last_layer)

        output_autoencoder = Dense(X.shape[1],
                                   activation='linear')(ae_last_layer)
        output_autoencoder = PReLU(name='autoencoder')(output_autoencoder)

        if self.bypass:
            supervised_input = Concatenate()([shared_dropout_layer,
                                              shared_last_layer])
        else:
            supervised_input = shared_last_layer

        output_sup = Dense(2, activation=softmax, name='sup')(supervised_input)

        model = Model(inputs=shared_input_layer,
                      output=[output_autoencoder, output_sup])
        model.compile(loss=['mean_squared_error', 'binary_crossentropy'],
                      loss_weights=[1. - self.alpha,
                                    self.alpha],
                      optimizer='rmsprop', metrics=['accuracy'],)

        self.hidden_model = Model(inputs=shared_input_layer,
                                  output=[shared_last_layer])
        return model

    def predict_proba(self, X):
        return self.network.predict(X)[1]

    def predict(self, X):
        return np.argmax(self.network.predict(X)[1], axis=1)

    def get_hidden_representation(self, X):
        return self.hidden_model.predict(X)
