import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense , Attention
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset of images.

    Returns:
    A tuple of two numpy arrays, each with shape (n_samples, 28, 28, 1),
    representing the training and testing data respectively.
    """
    try:
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    try:
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None, None

    try:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    except Exception as e:
        print(f"Error resizing dataset: {e}")
        return None, None

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    return x_train, x_test


if __name__ == "__main__":
    load_dataset()

