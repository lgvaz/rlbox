import tensorflow as tf


# TODO: hardcoded batch_size
def slice_2nd_dim(tensor, indices, batch_size):
    first_dim_ids = tf.range(batch_size, dtype=indices.dtype)
    slices_ids = tf.stack((first_dim_ids, indices), axis=1)

    return tf.gather_nd(tensor, slices_ids)
