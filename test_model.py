import numpy as np
import tensorflow as tf
from model import DQN

# STATE_SHAPE = [84, 84, 1]
# NUM_ACTIONS = 3
# LEARNING_RATE = 1e-3

STATE_SHAPE = [8]
NUM_ACTIONS = 3
# A higher learning rate can be used for simple envs
LEARNING_RATE = 1e-2

model = DQN(STATE_SHAPE, NUM_ACTIONS, LEARNING_RATE)

fake_states = np.random.random([3] + STATE_SHAPE)
fake_target_states = np.random.random([3] + STATE_SHAPE)
fake_actions = np.array([0, 0, 0])
fake_rewards = np.array([100, 100, 100])
fake_dones = np.array([1, 1, 1])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

old_preds = model.predict(sess, fake_states)
print('Old predictions:\n', old_preds)
for _ in range(100):
    model.fit(sess,
            fake_states,
            fake_target_states,
            fake_actions,
            fake_rewards,
            fake_dones)
new_preds = model.predict(sess, fake_states)
print('New predictions:\n', new_preds)
