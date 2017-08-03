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

        # Model architecture 
        output = Dense(self.num_actions)(states)

        model = Model(inputs=[states, actions], outputs=output, name='predictions')
        model.compile(optimizer=SGD(self.lr), loss=mask_loss(actions, self.num_actions))

        return model

    def predict(self, states):
        fake_actions = np.zeros(len(states))
        return self.model.predict([states, fake_actions])

    def fit(self, states, actions, labels, epochs=1):
        self.model.fit([states, actions], labels, epochs=epochs)
    
fake_inputs = np.random.random((1, 4))
fake_labels = np.random.random((1, 1))
fake_labels = np.array([(5)])
# fake_actions = np.random.randint(0, 1, 2, np.int32)
fake_actions = np.array([0])

model = DQN(state_shape=(4,), num_actions=2, learning_rate=1e-1)

old_preds = model.predict(fake_inputs)

model.fit(fake_inputs, fake_actions, fake_labels, epochs=200)

# h = model.model.fit([fake_inputs, fake_actions], fake_labels, epochs=200)

new_preds = model.predict(fake_inputs)
print('Original: {}'.format(fake_labels))
print('Old preds: {}'.format(old_preds))
print('New preds {}'.format(new_preds))
