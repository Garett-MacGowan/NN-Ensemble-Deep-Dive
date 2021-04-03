from typing import List

import tensorflow_addons as tfa

from nn_ensemble.components.data.data import Data

import tensorflow as tf


def train_model_1(models: List[tf.keras.Sequential], data: Data) -> List[tf.keras.Sequential]:
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    for model in models:
        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        model.fit(data.training_dataset,
                  epochs=100,
                  validation_data=data.validation_dataset,
                  callbacks=[early_stopping_callback])
    return models


def train_model_2(model: tf.keras.Sequential, data: Data) -> tf.keras.Sequential:
    avg_callback = tfa.callbacks.AverageModelCheckpoint(filepath='interim/model_2_checkpoints/cp-{epoch:04d}.ckpt',
                                                        update_weights=True)
    optimizer = tf.keras.optimizers.SGD()
    optimizer = tfa.optimizers.SWA(optimizer)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.fit(data.training_dataset,
              epochs=100,
              validation_data=data.validation_dataset,
              callbacks=[avg_callback])
    return model
