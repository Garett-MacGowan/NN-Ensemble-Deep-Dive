import tensorflow as tf

from typing import List

from configparser import ConfigParser


def construct_base_model() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(784, activation='relu'),
        tf.keras.layers.Dense(784, activation='relu'),
        tf.keras.layers.Dense(784, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')]
    )


def construct_model_1(config: ConfigParser) -> List[tf.keras.Sequential]:
    models = []
    for i in range(int(config['Model']['n_models'])):
        models.append(construct_base_model())
    return models


def construct_model_2() -> tf.keras.Sequential:
    return construct_base_model()
