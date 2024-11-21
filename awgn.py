import tensorflow as tf
from tensorflow import keras
import numpy as np

def add_awgn(
    input_tensor: tf.Tensor,  
    noise_std: float = 0.1  
) -> tf.Tensor:  
    """
    Adds Additive White Gaussian Noise (AWGN) to the input tensor.
    
    Args:
        input_tensor: Input tensor of shape (batch_size, 7, 7, 16).
        noise_std: Standard deviation of the Gaussian noise (controls noise strength).
    
    Returns:
        Tensor with added AWGN.
    """
    assert input_tensor is not None, "Input tensor is null."
    assert len(input_tensor.shape) == 4, "Input tensor must have 4 dimensions."
    batch_size, height, width, channels = input_tensor.shape

    if batch_size is None:
        batch_size=1

    noise = tf.random.normal(
        shape=(batch_size, height, width, channels),
        mean=0.0,
        stddev=noise_std,
        dtype=tf.float32
    )
    noisy_tensor = input_tensor + noise
    
    return noisy_tensor

if __name__ == "__main__":
    batch_size = 1  
    encoded_tensor = tf.random.normal((batch_size, 7, 7, 16))  
    noisy_encoded_tensor = add_awgn(encoded_tensor, noise_std=0.1)
    print(encoded_tensor.shape)
    print(noisy_encoded_tensor.shape)
