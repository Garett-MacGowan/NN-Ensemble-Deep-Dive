from pathlib import Path

from typing import List

from glob import glob

import tensorflow as tf

from configparser import ConfigParser

from nn_ensemble.components.data.data import Data
from nn_ensemble.components.model.model import construct_model_1, construct_model_2
from nn_ensemble.components.model.train import train_model_1, train_model_2


class ModelOrchestrator:

    def __init__(self, config: ConfigParser):
        self.config = config

        self.model_1_save_path = Path(f'interim/{self.config["Model"]["version"]}_model_1')

        self.model_2_save_path = Path(f'interim/{self.config["Model"]["version"]}_model_2')
        self.model_2_checkpoint_path = Path(f'interim/{self.config["Model"]["version"]}_model_2_checkpoints')

    def orchestrate(self, data: Data):
        # Create model 1 (n models)
        model_1 = construct_model_1(config=self.config)
        train_model_1(config=self.config, models=model_1, data=data,
                      save_path=self.model_1_save_path)

        # Create model 2 (SWA model)
        model_2 = construct_model_2()
        train_model_2(config=self.config, model=model_2, data=data,
                      save_path=self.model_2_save_path,
                      checkpoint_path=self.model_2_checkpoint_path)

    def load_model_generator(self, model_version: str):
        """
        This function takes a model version, and returns an iterator which loads the model when called. Useful for
        iterating through larger than memory ensembles.
        Args:
            model_version: An integer representing the model we want to load. Options are below:
                model_1: For loading all the models in the ensemble
                model_2: For loading the SWA model
                model_2_checkpoints: For loading the checkpoints used to create SWA model

        Returns: Generator that loads the model components iteratively

        """
        if model_version == 'model_1':
            for model_path in glob(str(Path(self.model_1_save_path, f'model_1_component_*'))):
                yield tf.keras.models.load_model(model_path)
        elif model_version == 'model_2':
            yield tf.keras.models.load_model(self.model_2_save_path)
        elif model_version == 'model_2_checkpoints':
            for model_path in glob(str(Path(self.model_2_checkpoint_path, f'cp-*.ckpt'))):
                yield tf.keras.models.load_model(model_path)
