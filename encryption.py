import tensorflow as tf
def xor_encrypt_decrypt(data, key=123):
    return tf.bitwise.bitwise_xor(data, key)