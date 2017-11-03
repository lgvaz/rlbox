import numpy as np
from sklearn.utils import shuffle


class DataGenerator:
    def __init__(self, placeholders_and_data_dict):
        assert isinstance(placeholders_and_data_dict, dict)
        self.placeholders = list(placeholders_and_data_dict.keys())
        self.data = list(placeholders_and_data_dict.values())
        self.data_size = np.shape(self.data[0])[0]

    def fetch_batch_dict(self, batch_size):
        assert batch_size <= self.data_size, 'Batch size is larger than dataset'
        self.data = shuffle(*self.data)
        num_batches = self.data_size // batch_size

        for i_batch in range(num_batches):
            start = i_batch * batch_size
            end = start + batch_size

            yield {
                placeholder: data[start:end]
                for placeholder, data in zip(self.placeholders, self.data)
            }
