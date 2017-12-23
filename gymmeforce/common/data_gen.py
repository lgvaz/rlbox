import numpy as np
from sklearn.utils import shuffle


class DataGenerator:
    def __init__(self, placeholders_and_data_dict):
        assert isinstance(placeholders_and_data_dict, dict)
        self.placeholders_and_data = placeholders_and_data_dict

        self.data_to_shuffle = {
            key: value
            for key, value in self.placeholders_and_data.items() if not np.isscalar(value)
        }
        self.data_scalar = {
            key: value
            for key, value in self.placeholders_and_data.items() if np.isscalar(value)
        }

        self.data_size = len(next(iter(self.data_to_shuffle.values())))

    def fetch_batch_dict(self, batch_size):
        assert batch_size <= self.data_size, 'Batch size is larger than dataset'
        num_batches = self.data_size // batch_size

        shuffled_data = {
            key: value
            for key, value in zip(self.data_to_shuffle.keys(),
                                  shuffle(*self.data_to_shuffle.values()))
        }

        for i_batch in range(num_batches):
            start = i_batch * batch_size
            end = start + batch_size

            batch = {
                placeholder: data[start:end]
                for placeholder, data in shuffled_data.items()
            }
            batch.update(self.data_scalar)

            yield batch
