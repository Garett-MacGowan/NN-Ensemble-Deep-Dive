import argparse
from argparse import Namespace
from argparse import ArgumentParser


class Arguments:
    _parser: ArgumentParser = None

    def __init__(self):
        parser = argparse.ArgumentParser(description=__doc__)

        parser.add_argument("-data", help="Prep the data",
                            action="store_true")
        parser.add_argument("-train", help="Train the models",
                            action="store_true")
        parser.add_argument("-analyze", help="Analyze the results",
                            action="store_true")
        self._parser = parser

    def get_arguments(self, argv) -> Namespace:
        """
        This function returns an ArgumentParser object which is used for control flow
        Args:
            argv: Arguments from python sys.argv

        Returns: ArgumentParser with the passed arguments.

        """
        args: Namespace = self._parser.parse_args(args=argv)
        return args
