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


