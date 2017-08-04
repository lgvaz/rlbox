import numpy as np
from model import DQN

STATE_SHAPE = [84, 84, 1]
NUM_ACTIONS = 3
LEARNING_RATE = 1e-3

# STATE_SHAPE = [4]
# NUM_ACTIONS = 3
# # A higher learning rate can be used for simple envs
# LEARNING_RATE = 1e-2

for i_action in range(NUM_ACTIONS):

    model = DQN(STATE_SHAPE, NUM_ACTIONS, LEARNING_RATE)

    fake_inputs = np.random.random([NUM_ACTIONS] + STATE_SHAPE)
    fake_labels = np.arange(1, NUM_ACTIONS + 1) * 2
    fake_actions = np.array([i_action] * NUM_ACTIONS)

    old_preds = model.predict(fake_inputs)
    model.fit(fake_inputs, fake_actions, fake_labels, epochs=100, verbose=0)
    new_preds = model.predict(fake_inputs)

    print('\nAction {} | Original: {}'.format(i_action, fake_labels))
    print('Old preds:\n{}'.format(old_preds))
    print('New preds:\n{}'.format(new_preds))
