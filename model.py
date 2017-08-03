import numpy as np
import keras
from keras.models import Model
from keras.layers import Lambda, Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K


def mask_loss(actions):
    def loss(y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred)

    return loss

# class DQN:
#     ''' Network model from DeepMind '''
#     def __init__(self):
actions = Input(shape=(1,))
inputs = Input(shape=(4,))
x = Dense(16)(inputs)
output = Dense(1)(inputs)

model = Model(inputs=[inputs, actions], outputs=output, name='predictions')
model.compile(optimizer=Adam(1e-2), loss=[mask_loss(actions=actions)])

fake_inputs = np.random.random((2, 4))
fake_labels = np.random.random((2, 1))
fake_actions = np.random.randint(0, 1, 2)

h = model.fit([fake_inputs, fake_actions], fake_labels, epochs=100)

preds = model.predict([fake_inputs, fake_actions])
print('Original: {}'.format(fake_labels))
print('Predicted: {}'.format(preds))
