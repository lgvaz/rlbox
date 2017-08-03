import numpy as np
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
        r = y_true - y_pred
        return K.mean(onehot_actions * K.square(r))
        # return keras.losses.mean_squared_error(y_true, y_pred)

    return loss

# class DQN:
#     ''' Network model from DeepMind '''
#     def __init__(self):
actions = Input(shape=(1,), dtype='int32')
inputs = Input(shape=(4,))
x = Dense(16)(inputs)
output = Dense(NUM_ACTIONS)(inputs)

model = Model(inputs=[inputs, actions], outputs=output, name='predictions')
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
