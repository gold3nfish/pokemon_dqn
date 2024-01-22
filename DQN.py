
import tensorflow as tf
from tensorflow.keras import layers, Sequential

class DQN(tf.keras.Model):
    def __init__(self, action_size, input_shape=(84, 84, 1)):
        super(DQN, self).__init__()
        # Convolutional Neural Network for feature extraction
        self.model = Sequential([
            layers.Conv2D(32, 8, strides=4, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(64, 4, strides=2, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, strides=1, activation='relu'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(action_size, activation='linear')
        ])

    def call(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = state / 255.0  # Normalize pixel values
        return self.model(state)
