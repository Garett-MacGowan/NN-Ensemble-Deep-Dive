from typing import List

from pathlib import Path

from glob import glob

import tensorflow_addons as tfa

from configparser import ConfigParser

from nn_ensemble.components.data.data import Data

import tensorflow as tf


def train_model_1(config: ConfigParser,
                  models: List[tf.keras.Sequential],
                  data: Data,
                  save_path: Path) -> None:
    version = config['Model']['version']

    for model_num, model in enumerate(models):
        print(f'Training model: {model_num}')

        # If the model already exists, skip this iteration
        model_save_path = Path(save_path, f'model_1_component_{model_num}')
        if len(glob(str(model_save_path))) > 0:
            print(f'WARNING: Skipping model number {model_num}, saved model already exists:\n{model_save_path}')
            continue

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        ]
        # Only log the first model, since they're all the same architecture.
        if model_num == 0:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=f'logs/{version}_model_1_{model_num}',
                    profile_batch='100, 110',
                    histogram_freq=1,
                    update_freq='batch'
                )
            )
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=float(config['Model']['learning_rate'])),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        model.fit(data.training_dataset,
                  epochs=1000,
                  validation_data=data.validation_dataset,
                  callbacks=callbacks)

        # Save the model
        model.save(model_save_path)
        # Remove the model from memory, since OOM might occur.
        del model


def train_model_2(config: ConfigParser,
                  model: tf.keras.Sequential,
                  data: Data,
                  save_path: Path,
                  checkpoint_path: Path) -> None:
    version = config['Model']['version']

    callbacks = [
        tfa.callbacks.AverageModelCheckpoint(filepath=str(checkpoint_path) + '/cp-{epoch:04d}.ckpt',
                                             update_weights=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/{version}_model_2',
            profile_batch='100, 110',
            histogram_freq=1,
            update_freq='batch'
        )
    ]
    optimizer = tf.keras.optimizers.SGD(learning_rate=float(config['Model']['learning_rate']))
    # 35 below obtained by inspecting the epoch at which convergence occurred on validation set with TensorBoard.
    optimizer = tfa.optimizers.SWA(optimizer, start_averaging=35, average_period=int(config['Model']['n_models']))

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.fit(data.training_dataset,
              epochs=1000,
              validation_data=data.validation_dataset,
              callbacks=callbacks)

    # Save the model
    model.save(save_path)
    # Remove the model from memory, since OOM might occur.
    del model
