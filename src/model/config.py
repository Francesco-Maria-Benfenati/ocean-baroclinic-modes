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
        # check if the file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        self.config_file = config_path
        self.__read_config_file__(config_path)
        self.__from_dict_to_attrs__()

    def load_config_file(self, config_path: str) -> None:
        """
        Overwrite an existing configuration with a new one
        :param config_path: path to the config file
        :return: returns nothing
        """
        # check if the file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        self.config_file = config_path
        self.__read_config_file__(config_path)
        self.__from_dict_to_attrs__()

    def __read_config_file__(self, config_path: str) -> None:
        """
        Inner private function to read the file
        :param config_path: path to the config file
        :return: returns nothing
        """
        with open(config_path, mode="r") as f:
            self.config_dict = tomllib.loads(f.read())
        return self.config_dict

    def __from_dict_to_attrs__(self) -> None:
        """
        Set attributes out of config dictionary.
        """
        for k, v in self.config_dict.items():
            if hasattr(self, k):
                delattr(self, k)  # Overwrite attribute
            attr = getattr(self, k, v)
            setattr(self, k, AttrDict(attr))


class AttrDict(dict):
    """
    Dictionary subclass whose entries can be accessed by attributes
    """

    def __init__(self, *args, **kwargs) -> dict:
        def from_nested_dict(data):
            """Construct nested AttrDicts from nested dictionaries."""
            if not isinstance(data, dict):
                return data
            else:
                return AttrDict({key: from_nested_dict(data[key]) for key in data})

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])


if __name__ == "__main__":
    import numpy as np
    import sys

    file = "./config.toml"
    config = Config(file)

    print(config.config_file)
    print(config.config_dict)

    from pprint import pprint

    key = "domain"
    value = config.config_dict[key]

    print(value)
    pprint(value)
    print("Domain mean latitude:")
    # Access to config values as dictionary items
    print(np.mean(np.array(config.config_dict["domain"]["lat"])))
    # Access to config values as attributes
    print(np.mean(np.array(config.domain.lat)))
    print(config.input)
    print("Output file name: ", config.output.filename)
    config.domain.clear()
    try:
        print(np.mean(np.array(config.domain.lat)))
    except AttributeError:
        print("Domain values cleared")
    config.input.clear()
    config.load_config_file(file)
    print(config.input.oce.vars)
    print(config.input.oce.dims)
    # Datetime format
    print(type(config.domain.datetime[0]), config.domain.datetime[:])
    print(np.datetime64(config.domain.datetime[0]))
