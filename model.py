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
    ''' Network model from DeepMind '''
    def __init__(self, state_shape, num_actions, learning_rate):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = learning_rate

        self.model = self._build_model()

    def _build_model(self):
        # Model inputs
        states = Input(shape=(self.state_shape))
        actions = Input(shape=(1,), dtype='int32')

        # # Model architecture 
        # net = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(states)
        # net = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(net)
        # net = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(net)

        net = states
        net = Flatten()(net)
        # net = Dense(512, activation='relu')(net)

        output = Dense(self.num_actions)(net)

        model = Model(inputs=[states, actions], outputs=output, name='predictions')
        model.compile(optimizer=Adam(self.lr), loss=mask_loss(actions, self.num_actions))

        return model

    def predict(self, states):
        fake_actions = np.zeros(len(states))
        return self.model.predict([states, fake_actions])

    def fit(self, states, actions, labels, epochs=1, verbose=1):
        self.model.fit([states, actions], labels, epochs=epochs, verbose=verbose)

