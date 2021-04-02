from configparser import ConfigParser

from matplotlib import pyplot as plt

from typing import Tuple

import tensorflow as tf

import numpy as np


class Data:
    training_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset
    testing_dataset: tf.data.Dataset

    def __init__(self, config: ConfigParser):
        self.config = config

    def prep_data(self) -> None:
        train_data, test_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')

        train_data, validation_data = self._train_validate_split(train_data)

        self.training_dataset = self._prep_tf_dataset(train_data)
        self.validation_dataset = self._prep_tf_dataset(validation_data)
        self.testing_dataset = self._prep_tf_dataset(test_data)

    def _train_validate_split(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                                                  Tuple[np.ndarray, np.ndarray]]:
        """
        This function takes in a tuple of independent and dependent variables and splits the data into two tuples of
        independent and dependent variables, based on a split defined in the config.
        Args:
            data: A tuple of independent and dependent variables (most likely training data to be split into training/
            validation)

        Returns: A tuple of tuples containing the original data, split by the ratio defined in the config.

        """
        split_points = float(self.config['Data']['train_validate_split'])
        data_len = data[0].shape[0]

        training_data_slice = (data[0][:int(split_points * data_len), :],
                               data[1][:int(split_points * data_len)])
        validation_data_slice = (data[0][int(split_points * data_len) + 1:, :],
                                 data[1][int(split_points * data_len) + 1:])
        return training_data_slice, validation_data_slice

    @staticmethod
    def _prep_tf_dataset(data: Tuple[np.ndarray, np.ndarray]) -> tf.data.Dataset:
        """
        This function creates a tensorflow dataset given a tuple of numpy arrays with independent and dependent
        variables.
        Args:
            data: A tuple of numpy arrays with independent and dependent variables.

        Returns: A tensorflow dataset.

        """

        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255.0, label

        return tf.data.Dataset.from_tensor_slices(data)\
            .map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .shuffle(buffer_size=data[0].shape[0], seed=0)\
            .batch(32)

    @staticmethod
    def visualize_data(dataset: tf.data.Dataset) -> None:
        """
        This function visualizes the data in the given dataset.
        Args:
            dataset: A dataset to visualize

        Returns:

        """
        for x, y in dataset.take(1):
            image = x[0].numpy()
            plt.imshow(image, cmap='gray')
            plt.title(label=f"Dependent variable (class label): {y[0].numpy()}")
            plt.show()
