import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from attention import AttentionLayer

class Encoder(Model):
    def __init__(self):
        """
        Initialize the Encoder model.

        Args:
            None

        Returns:
            None
        """
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.maxpool1 = layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.maxpool2 = layers.MaxPooling2D((2, 2), padding='same')
        self.attention_layer1 = AttentionLayer() 
        self.attention_layer2 = AttentionLayer()
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the encoder model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Output tensor after processing through convolutional, attention, and pooling layers.
        """
        x = self.conv1(inputs)
        # x = self.attention_layer1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # x = self.attention_layer2(x)
        x = self.maxpool2(x)
        return x
