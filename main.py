from autoencoder import Autoencoder
from dataset import load_dataset
import matplotlib.pyplot as plt
from awgn import add_awgn
import tensorflow as tf 
from encryption import xor_encrypt_decrypt

def main(autoencoder: Autoencoder) -> tf.keras.callbacks.History:
    """
    Train the autoencoder model.

    Args:
        autoencoder (Autoencoder): Instance of the Autoencoder model to be trained.

    Returns:
        tf.keras.callbacks.History: The history object containing training metrics.
    """
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    print(autoencoder.summary())
    x_train_encrypted=xor_encrypt_decrypt(x_train)
    x_test_encrypted=xor_encrypt_decrypt(x_test)
    history = autoencoder.fit(x_train_encrypted, x_train_encrypted, epochs=20, batch_size=256, validation_data=(x_test_encrypted, x_test_encrypted))
    return history
def plotting(history: tf.keras.callbacks.History) -> None:
    """
    Plot the training and validation loss curves.

    Args:
        history (tf.keras.callbacks.History): The history object containing the training and validation metrics.

    Returns:
        None
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def testing() -> None:
    """
    Test the autoencoder model.

    Args:
        None

    Returns:
        None
    """
    x_test_encrypted=xor_encrypt_decrypt(x_test)
    encoded_imgs = autoencoder.encoder.predict(x_test_encrypted)
    noisy_images=add_awgn(encoded_imgs, noise_std=0.1)
    decoded_imgs = autoencoder.decoder.predict(noisy_images)

    n = 10  
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        ax = plt.subplot(3, n, i + n + 1)
        plt.imshow(encoded_imgs[i].reshape(7, 7 * 16), cmap='gray')  
        plt.title("Encoded")
        plt.axis('off')

        ax = plt.subplot(3, n, i + 2 * n + 1)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()

if __name__=="__main__":
    x_train, x_test=load_dataset()
    print(x_train.shape, x_test.shape)
    autoencoder = Autoencoder()
    history=main(autoencoder)
    plotting(history)
    testing()