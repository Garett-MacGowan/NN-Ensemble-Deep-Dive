import os
from configparser import ConfigParser


class __Config:
    _config_path = '{}.config'

    _env_var = 'NN_ENSEMBLE_CONFIGURATION'

    _error_no_environment_variable = f"""

        Cannot continue without configuration. 

        To set a configuration, add the environment variable below.

        '{_env_var}'

        The value of the environment variable should be set to the config suffix phrase. e.g. 'sample' for linking
        to the sample.config file provided.
        """

    _error_no_config_file = "The active profile is set to '{}' but no file named '{}' exists in '{}'"

    @property
    def get(self) -> ConfigParser:
        """
        Gets a ConfigParser object, assuming the checks pass.
        Returns: ConfigParser object containing the configurations.

        """
        assert self._env_var in os.environ, self._error_no_environment_variable

        profile = os.environ[self._env_var]
        profile_config_path = self._config_path.format(profile)

        assert os.path.isfile(profile_config_path), self._error_no_config_file.format(
            profile, profile_config_path, os.getcwd())

        cp = ConfigParser()
        cp.read(self._config_path.format(profile))
        return cp


def get() -> ConfigParser:
    return __Config().get
