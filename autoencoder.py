import tensorflow as tf
from tensorflow.keras import layers, models
from encoder import Encoder
from decoder import Decoder
from awgn import add_awgn


class Autoencoder(tf.keras.Model):
    def __init__(self) -> None:
        """
        Initialize the Autoencoder model.

        Args:
            None

        Returns:
            None
        """
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()  
        self.decoder = Decoder()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the Autoencoder model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Output tensor of shape (batch_size, height, width, channels).
        """
        encoded_img = self.encoder(inputs)
        
        noisy_image = add_awgn(encoded_img, noise_std=0.1)  

        decoded_img = self.decoder(noisy_image)
        
        return decoded_img
