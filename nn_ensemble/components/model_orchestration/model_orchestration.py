from pathlib import Path

from typing import List

from glob import glob

import tensorflow as tf

from configparser import ConfigParser

from nn_ensemble.components.data.data import Data
from nn_ensemble.components.model.model import construct_model_1, construct_model_2
from nn_ensemble.components.model.train import train_model_1, train_model_2


class ModelOrchestrator:
    model_1: List[tf.keras.Sequential]
    model_2: tf.keras.Sequential

    model_1_save_path = Path('nn_ensemble/iterim/model_1')
    model_2_save_path = Path('nn_ensemble/iterim/model_2')

    def orchestrate(self, data: Data, config: ConfigParser):
        # Create model 1 (n models)
        self.model_1 = construct_model_1(config=config)
        self.model_1 = train_model_1(models=self.model_1, data=data)
        self._save_model_1()

        # Create model 2 (SWA model)
        self.model_2 = construct_model_2()
        self.model_2 = train_model_2(model=self.model_2, data=data)
        self._save_model_2()

    def _save_model_1(self):
        for i, model in enumerate(self.model_1):
            model.save(Path(self.model_1_save_path, f'model_1_component_{i}'))

    def _save_model_2(self):
        self.model_2.save(self.model_2_save_path)

    def load_models(self):
        model_1 = []
        for model_path in glob(Path(self.model_1_save_path, f'model_1_component_*')):
            model_1.append(tf.keras.models.load_model(model_path))

        model_2 = tf.keras.models.load_model(self.model_2_save_path)

        self.model_1 = model_1
        self.model_2 = model_2
