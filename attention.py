import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Multiply, Add, Permute

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        """
        Initializes the AttentionLayer class.

        Raises:
            ValueError: If the input shape is not defined.
        """
        super(AttentionLayer, self).__init__()

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the Conv2D layers for the attention mechanism.

        Args:
            input_shape: The shape of the input tensor, used to determine the
                number of filters in the Conv2D layers.

        Returns:
            None
        """
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.num_channels = input_shape[-1]

        self.spatial_conv = Conv2D(self.num_channels, (1, 1), activation='sigmoid')
        self.channel_conv = Conv2D(self.num_channels, (1, 1), activation='sigmoid')
        self.joint_channel_conv = Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply attention mechanisms to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output tensor with attention applied.
        """
        input = inputs
        
        spatial_attention = self.spatial_conv(inputs)
        spatial_out = Multiply()([inputs, spatial_attention])

       
        x_permuted = Permute((3, 1, 2))(inputs)
        channel_attention = self.channel_conv(x_permuted)
        channel_out = Permute((2, 3, 1))(Multiply()([x_permuted, channel_attention]))

        x_expanded = tf.expand_dims(inputs, axis=-1)
        joint_channel_attention = self.joint_channel_conv(x_expanded)
        joint_channel_out = tf.squeeze(joint_channel_attention, axis=-1)
        joint_channel_out = Multiply()([inputs, joint_channel_out])


        out = Add()([spatial_out, channel_out, joint_channel_out])
        return out
