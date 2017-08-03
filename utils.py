import tensorflow as tf
from keras import backend as K


def mask_loss(actions, num_actions):
    '''
    The target only corresponds to the selected action,
    so the error must be masked to only inclued the loss from the chosen action
    '''
    def loss(y_true, y_pred):
        onehot_actions = K.one_hot(actions, num_actions)
        errors = huber_loss(y_true, y_pred)
        return K.mean(onehot_actions * errors)
    return loss

def huber_loss(y_true, y_pred, delta=1):
    '''
    Hubber loss is less sensitive to outliers
    https://en.wikipedia.org/wiki/Huber_loss
    '''
    error = y_true - y_pred
    condition = K.abs(error) <= delta
    squared_error =  0.5 * K.square(error)
    linear_error = delta * (K.abs(error) - 0.5 * delta)
    return tf.where(condition, squared_error, linear_error)
