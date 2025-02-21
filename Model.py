import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
import gc
from DataFlow import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, TRAINING_DATASET_DIR, TESTING_DATASET_DIR

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_image_translation_model():
    input1 = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    input2 = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))

    merged_input = layers.Concatenate()([input1, input2])

    x = layers.Conv2D(64, (3, 3), activation='tanh', padding='same')(merged_input)

    x = layers.Conv2D(128, (3, 3), activation='tanh', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='tanh', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), activation='tanh', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.SeparableConv2D(512, (3, 3), activation='tanh', padding='same')(x)
    x = layers.Reshape((1, IMG_HEIGHT // 8, IMG_WIDTH // 8, 512))(x)  # Prepare for ConvLSTM2D

    x = layers.ConvLSTM2D(512, (3, 3), activation='tanh', padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(512, (3, 3), activation='tanh', padding='same', return_sequences=False)(x)

    x = layers.Conv2D(512, (3, 3), activation='tanh', padding='same')(x)
    # x = layers.Conv2DTranspose(512, (3, 3), activation='tanh', padding='same')(x)

    x = layers.Conv2DTranspose(256, (3, 3), strides=2, activation='tanh', padding='same')(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='tanh', padding='same')(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='tanh', padding='same')(x)

    output = layers.Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')(x)
    model = models.Model(inputs=[input1, input2], outputs=output)
    return model

model = create_image_translation_model()
print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

data_train_paths = sorted(
    [os.path.join(TRAINING_DATASET_DIR, fname) for fname in os.listdir(TRAINING_DATASET_DIR) if fname.endswith(".npz")])
data_test_paths = sorted(
    [os.path.join(TESTING_DATASET_DIR, fname) for fname in os.listdir(TESTING_DATASET_DIR) if fname.endswith(".npz")])


for i in range(0, len(data_train_paths)):
    train_data = np.load(data_train_paths[i])["train"]
    test_data = np.load(data_test_paths[i])["test"]
    print(i)

    X1, X2 = zip(*train_data)
    X1, X2 = np.array(X1), np.array(X2)
    Y = np.array(test_data)

    X1 = X1[:min(len(X1), len(Y))]
    X2 = X2[:min(len(X1), len(Y))]
    Y = Y[:min(len(X1), len(Y))]

    if len(X1) != len(Y):
        print(data_train_paths[i])
        print(len(X1), len(Y))
        continue

    X1, X2, Y = shuffle(X1, X2, Y, random_state=42)

    model.fit([X1, X2], Y, epochs=10, batch_size=8, validation_split=0.1)

    K.clear_session()

    gc.collect()

    model.save(f"normalized_image_translation_model_LSTM_less{i}_high_gpu", save_format="tf")
