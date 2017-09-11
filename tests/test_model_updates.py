import numpy as np
import tensorflow as tf
from model import DQNModel


# TODO: Change print statements to asserts
STATE_SHAPE = [84, 84, 4]
NUM_ACTIONS = 3
LEARNING_RATE = 1e-3
fake_states = np.random.randint(0, 255, size=[3] + STATE_SHAPE, dtype=np.uint8)
fake_target_states = np.random.randint(0, 255, size=[3] + STATE_SHAPE, dtype=np.uint8)

# STATE_SHAPE = [8]
# NUM_ACTIONS = 3
# # A higher learning rate can be used for simple envs
# LEARNING_RATE = 1e-2
# fake_states = np.random.random([3] + STATE_SHAPE)
# fake_target_states = np.random.random([3] + STATE_SHAPE)


fake_rewards = np.array([100, 100, 100])
fake_dones = np.array([1, 1, 1])

print('Testing action optimization process')
for i_action in range(NUM_ACTIONS):
    fake_actions = np.array(3 * [i_action])

    tf.reset_default_graph()
    model = DQNModel(STATE_SHAPE, NUM_ACTIONS)

    print('Optimizing for action', i_action)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        old_preds = model.predict(sess, fake_states)
        print('Old predictions:\n', old_preds)
        for _ in range(100):
            model.train(sess,
                        LEARNING_RATE,
                        fake_states,
                        fake_target_states,
                        fake_actions,
                        fake_rewards,
                        fake_dones)
        new_preds = model.predict(sess, fake_states)
        print('New predictions:\n', new_preds)

print('Testing target update process')
tf.reset_default_graph()
model = DQNModel(STATE_SHAPE, NUM_ACTIONS)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    online_preds = model.predict(sess, fake_states)
    old_target_preds = model.target_predict(sess, fake_states)
    model.update_target_net(sess)
    new_target_preds = model.target_predict(sess, fake_states)

    print('Online predictions:\n', online_preds)
    print('Old target predictions:\n', old_target_preds)
    print('New target predictions:\n', new_target_preds)
