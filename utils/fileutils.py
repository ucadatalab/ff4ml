# coding: utf-8

import glob
import yaml


def load_config(path_to_config_file):
    """
        Load config file

        Parameters
        ----------
        path_to_config_file: str
            Path to *.yaml file

        Return
        ------
        configuration:
            The yaml file loaded

        Raise
        -----
        e: Exception
             If something was wrong.


    """
    try:
        with open(path_to_config_file) as f:
            configuration = yaml.load(f)
    except Exception as e:
        print("There was an ERROR loading the configuration file: " + path_to_config_file)
        raise e

    return configuration


