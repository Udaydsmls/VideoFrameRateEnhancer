import math

import tensorflow as tf
from tensorflow.keras import layers, models


def create_image_translation_model(img_height: int, img_width: int, num_channels: int) -> tf.keras.models.Model:
    """
    Builds an image translation model with ConvLSTM layers for spatiotemporal processing.
    """
    input1 = layers.Input(shape=(img_height, img_width, num_channels))
    input2 = layers.Input(shape=(img_height, img_width, num_channels))

    merged_input = layers.Concatenate()([input1, input2])

    x = layers.Conv2D(64, (3, 3), activation='tanh', padding='same')(merged_input)

    x = layers.Conv2D(128, (3, 3), activation='tanh', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(256, (3, 3), activation='tanh', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(512, (3, 3), activation='tanh', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.SeparableConv2D(512, (3, 3), activation='tanh', padding='same')(x)
    x = layers.Reshape((1, math.ceil(img_height / 8), math.ceil(img_width / 8), 512))(x)  # Prepare for ConvLSTM2D

    x = layers.ConvLSTM2D(512, (3, 3), activation='tanh', padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(512, (3, 3), activation='tanh', padding='same', return_sequences=False)(x)

    x = layers.Conv2DTranspose(256, (3, 3), strides=2, activation='tanh', padding='same')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='tanh', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='tanh', padding='same')(x)
    output = layers.Conv2DTranspose(num_channels, (3, 3), activation='tanh', padding='same')(x)
    output = layers.Cropping2D(cropping=((output.shape[1] - img_height, 0), (output.shape[2] - img_width, 0)))(output)

    return models.Model(inputs=[input1, input2], outputs=output)
