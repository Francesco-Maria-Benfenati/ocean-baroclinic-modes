# try to import both for compatibility with older versions of python ( tomlibb for pytohn 3.11 and above, toml for python 3.10 and below)
try:
    import tomllib
except ImportError:
    import toml as tomllib

import os


class Config:
    """
    Class to read a config file
    """

    def __init__(self, config_path: str):
        """
        Create a Config object from a config file
        :param config_path: path to the config file
        :return: returns nothing
        """
        self.config_file = config_path
        # check if the file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        self.__read_config_file__(config_path)

    def load_config_file(self, config_path: str) -> None:
        """
        Overwrite an existing configuration with a new one
        :param config_path: path to the config file
        :return: returns nothing
        """
        self.config_file = config_path
        self.__read_config_file__(config_path)

    def __read_config_file__(self, config_path: str) -> None:
        """
        Inner private function to read the file
        :param config_path: path to the config file
        :return: returns nothing
        """
        with open(config_path, mode="r") as f:
            self.config_dict = tomllib.loads(f.read())


if __name__ == "__main__":
    import sys
    import pandas as pd
    import sys

    config = Config(sys.argv[1])
    print(config.config_file)
    print(config.config_dict)

    from pprint import pprint

    key = ""
    value = config.config_dict[key]

    print(value)
    pprint(value)
