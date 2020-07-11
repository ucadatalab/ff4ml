# coding: utf-8

"""
    :mod:`fileutils`
    ===========================================================================
    :synopsis: Utility class to manage file operaations
    :author: UCADatalab - http://datalab.uca.es, NESG (Network Engineering & Security Group) - https://nesg.ugr.es
    :contact: ignacio.diaz@uca.es, roberto.magan@uca.es, rmagan@ugr.es
    :organization: University of Cádiz
    :project: ff4ml (Free Framework for Machine Learning)
    :since: 0.0.1
"""

import yaml
import json

yaml.warnings({'YAMLLoadWarning': False})

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


def params_to_json(bp, path_to_json):
    """
    Dumps model parameters to a JSON file.

    Parameters
    ----------
    bp: dict
        Best selected parameters.
    path_to_json: str
        Path to the json file

    Raise
    -----
    e: Exception
        If something was wrong.

    """
    try:
        with open(path_to_json, "w") as fi:
            json.dump(bp, fi)
    except Exception as e:
        print("There was an ERROR writing on file " + path_to_json)
        raise e


def print_config(config):
    """
    Outputs all the configuration parameters and values got from the config file.

    :param config: dict
        Contains all the parameters
    """
    print(json.dumps(config, indent=4))

