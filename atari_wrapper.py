import gym
import cv2
import numpy as np

class AtariWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def _preprocess_frame(self, frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        img *= [0.299, 0.587, 0.114]
        resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        return self._preprocess_frame(state), reward, done, info

    def reset(self):
        state = env.reset()

        return self._preprocess_frame(state)


env = AtariWrapper('Pong-v0')
