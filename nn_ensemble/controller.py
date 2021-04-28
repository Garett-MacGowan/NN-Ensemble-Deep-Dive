from typing import List, Optional
from configparser import ConfigParser

from argparse import Namespace

from nn_ensemble.components.data.data import Data
from nn_ensemble.components.model_orchestration.model_orchestration import ModelOrchestrator

from nn_ensemble.components.analysis.analyze import Analysis

from nn_ensemble.configurations.arguments import Arguments

from datetime import datetime


class Controller:

    def execute(self, argv: List[str], config: ConfigParser) -> None:
        """
        This function executes the control flow for the Neural Network Ensemble Deep Dive.
        Args:
            argv: The arguments passed to the main.py entry point.
            config: The configuration from the .config file.

        Returns:

        """
        # Tracking execution wall time.
        start_time: datetime = datetime.now()

        arguments: Arguments = Arguments()
        args: Namespace = arguments.get_arguments(argv)

        data: Optional[Data] = None
        if args.data:
            data: Data = Data(config=config)
            data.prep_data()
            data.visualize_data(data.training_dataset)

        model_orchestrator: ModelOrchestrator = ModelOrchestrator(config=config)
        if args.train:
            model_orchestrator.orchestrate(data)

        if args.analyze:
            analysis: Analysis = Analysis(config=config, data=data, model_orchestrator=model_orchestrator)
            analysis.analyze()

        end_time: datetime = datetime.now()
        print(f'Neural Network Ensemble Deep Dive completed in {end_time - start_time}')
