# -- ici nous definissiosn les generateurs, discriminateur et le train--

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

#--generateurs------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=100),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(28 * 28 * 1, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

def build_generator_best():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_dim=100),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model



#--disrimiteurs----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_discriminator_best():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model



#--compilateur/entrainement----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def train_gan(gan, generator, discriminator, dataset, epochs=5000, batch_size=64):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_images = generator.predict(noise)

        real_images = dataset[np.random.randint(0, dataset.shape[0], batch_size)]
        combined_images = np.concatenate([real_images, generated_images])
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(combined_images, labels)

        noise = np.random.normal(0, 1, (batch_size, 100))
        misleading_labels = np.ones((batch_size, 1))
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, misleading_labels)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")


#--visualisation------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_generated_images(generator, examples=16, dim=(4, 4), figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Re-normalisation

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
