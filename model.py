import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Lambda, Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K

NUM_ACTIONS = 2

def mask_loss(actions):
    def loss(y_true, y_pred):
        onehot_actions = K.one_hot(actions, NUM_ACTIONS)
        errors = huber_loss(y_true, y_pred)
        return K.mean(onehot_actions * errors)
        # return keras.losses.mean_squared_error(y_true, y_pred)

    return loss

def huber_loss(y_true, y_pred, delta=1):
    '''
    Hubber loss is less sensitive to outliers
    https://en.wikipedia.org/wiki/Huber_loss
    '''
    error = y_true - y_pred
    condition = K.abs(error) <= delta
    squared_error =  0.5 * K.square(error)
    linear_error = delta * (K.abs(error) - 0.5 * delta)
    return tf.where(condition, squared_error, linear_error)

# class DQN:
#     ''' Network model from DeepMind '''
#     def __init__(self):
# Model inputs
actions = Input(shape=(1,), dtype='int32')
states = Input(shape=(4,))

# Model architecture 
output = Dense(NUM_ACTIONS)(states)

model = Model(inputs=[states, actions], outputs=output, name='predictions')
model.compile(optimizer=SGD(1e-1), loss=mask_loss(actions=actions))

fake_inputs = np.random.random((1, 4))
fake_labels = np.random.random((1, 1))
fake_labels = np.array([(100)])
# fake_actions = np.random.randint(0, 1, 2, np.int32)
fake_actions = np.array([0])

old_preds = model.predict([fake_inputs, fake_actions])

h = model.fit([fake_inputs, fake_actions], fake_labels, epochs=200)

new_preds = model.predict([fake_inputs, fake_actions])
print('Original: {}'.format(fake_labels))
print('Old preds: {}'.format(old_preds))
print('New preds {}'.format(new_preds))
