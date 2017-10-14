import time
import numpy as np
from collections import defaultdict


class Logger:
    def __init__(self, debug=False):
        self.debug = debug
        self.logs = defaultdict(list)
        self.precision = dict()
        self.time = time.time()
        self.tf_scalar_summary_writer = None
        self.sess = None

    def add_tf_writer(self, sess, tf_scalar_summary_writer):
        self.sess = sess
        self.tf_scalar_summary_writer = tf_scalar_summary_writer

    def add_log(self, name, value, precision=2):
        self.logs[name].append(value)
        self.precision[name] = precision

    def add_debug(self, name, value, precision=2):
        if self.debug:
            self.add_log(name, value, precision)

    def log(self, header=None):
        ''' Write the mean of the values added to each key and clear previous values '''
        avg_dict = {key: '{:.{prec}f}'.format(np.mean(value), prec=self.precision[key])
                    for key, value in self.logs.items()}

        # Log to the console
        self.logs = defaultdict(list)
        print_table(avg_dict, header)

        # Write tensorflow summary
        if self.tf_scalar_summary_writer is not None:
            for key, value in avg_dict.items():
                self.tf_scalar_summary_writer(self.sess, key, float(value))

    def timeit(self, steps):
        new_time = time.time()
        steps_sec = steps / (new_time - self.time)
        self.add_log('Steps/Second', steps_sec)
        self.time = new_time


def print_table(tags_and_values_dict, header=None, width=42):
    '''
    Print a pretty table =)
    Expects keys and values of dict to be a string
    '''

    tags_maxlen = max(len(tag) for tag in tags_and_values_dict)
    values_maxlen = max(len(value) for value in tags_and_values_dict.values())

    max_width = max(width, tags_maxlen + values_maxlen)

    print()
    if header:
        print(header)
    print((2 + max_width) * '-')
    for tag, value in tags_and_values_dict.items():
        num_spaces = 2 + values_maxlen - len(value)
        string_right = '{:{n}}{}'.format('|', value, n=num_spaces)
        num_spaces = 2 + max_width - len(tag) - len(string_right)
        print(''.join((tag, ' ' * num_spaces, string_right)))
    print((2 + max_width) * '-')
