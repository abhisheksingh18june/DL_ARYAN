import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from attention import AttentionLayer

class Decoder(Model):
    def __init__(self):
        """
        Initialize the Decoder model.

        Args:
            None

        Returns:
            None
        """
        super(Decoder, self).__init__()
        
        self.conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.upsample1 = layers.UpSampling2D((2, 2))
        
        self.conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.upsample2 = layers.UpSampling2D((2, 2))
        
        self.conv3 = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        
        self.attention_layer1 = AttentionLayer()  
        self.attention_layer2 = AttentionLayer()


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the decoder model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Output tensor of shape (batch_size, height, width, channels).
        """
        x = self.conv1(inputs)
        # x = self.attention_layer1(x)
        x = self.upsample1(x)
        
        x = self.conv2(x)
        # x = self.attention_layer2(x)  
        x = self.upsample2(x)
        
        x = self.conv3(x)
        
        return x

