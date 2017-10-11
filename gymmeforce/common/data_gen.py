import numpy as np
from sklearn.utils import shuffle


class DataGenerator:
    def __init__(self, *args):
        self.data = args
        self.data_size = np.shape(self.data[0])[0]

    def iterate_once(self, batch_size):
        self.data = shuffle(*self.data)
        assert batch_size <= self.data_size, 'Batch size is larger than dataset'
        num_batches = self.data_size // batch_size

        for i_batch in range(num_batches):
            start = i_batch * batch_size
            end = start + batch_size

            yield [data[start : end] for data in self.data]
