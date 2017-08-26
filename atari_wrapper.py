import gym
from gym import Wrapper
import cv2
import numpy as np

class AtariWrapper(Wrapper):
    def __init__(self, env):
        super(AtariWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84))

    def _preprocess_frame(self, frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_LINEAR)
        x_t = resized_screen[18:102, :]

        return x_t.astype(np.uint8)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self._preprocess_frame(state)

        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self._noops()
        state = self._preprocess_frame(state)

        return state

    def _noops(self, num_noops=20):
        ''' Do nothing for num_oops frames '''
        for _ in range(num_noops):
            state, _, _, _ = self.env.step(0)

        return state
