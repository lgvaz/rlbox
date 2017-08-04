import numpy as np
import keras
from keras.models import Model
from keras.layers import Lambda, Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K
from utils import *

NUM_ACTIONS = 2

class DQN:
    def __init__(self, state_shape, num_actions, learning_rate):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = learning_rate

        if len(state_shape) == 3:
            self.model = self._build_deepmind_model()
        elif len(state_shape) == 1:
            self.model = self._build_dense_model()
        else:
            raise ValueError('state_shape not supported')

    def _build_deepmind_model(self):
        ''' Network model from DeepMind '''
        # Model inputs
        states = Input(shape=(self.state_shape))
        actions = Input(shape=(1,), dtype='int32')

        # Model architecture
        net = states
        # Convolutional layers
        net = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(net)
        net = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(net)
        net = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(net)
        net = Flatten()(net)

        # Dense layers
        net = Dense(512, activation='relu')(net)

        output = Dense(self.num_actions)(net)

        model = Model(inputs=[states, actions], outputs=output, name='predictions')
        model.compile(optimizer=Adam(self.lr), loss=mask_loss(actions, self.num_actions))

        return model

    def _build_dense_model(self):
        ''' Simple fully connected model '''
        # Model inputs
        states = Input(shape=(self.state_shape))
        actions = Input(shape=(1,), dtype='int32')

        # Model architecture
        net = states
        net = Dense(64, activation='relu')(net)
        # net = Dense(64, activation='relu')(net)
        output = Dense(self.num_actions)(net)

        model = Model(inputs=[states, actions], outputs=output, name='predictions')
        model.compile(optimizer=Adam(self.lr), loss=mask_loss(actions, self.num_actions))

        return model

    def predict(self, states):
        fake_actions = np.zeros(len(states))
        return self.model.predict([states, fake_actions])

    def fit(self, states, actions, labels, epochs=1, verbose=1):
        self.model.fit([states, actions], labels, epochs=epochs, verbose=verbose)

