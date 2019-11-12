# -*- coding: utf-8 -*-
"""
    :mod:`main`
    ===========================================================================
    :synopsis: Main class
    :author: UCADatalab - http://datalab.uca.es, NESG (Network Engineering & Security Group) - https://nesg.ugr.es
    :contact: roberto.magan@uca.es, rmagan@ugr.es
    :organization: University of Cádiz
    :project: ff4ml (Free Framework for Machine Learning)
    :since: 0.0.1
"""

import argparse

def getArguments():
    """
    Function to get input arguments from configuration file
    :return: args
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''ff4ml (Free Framework for Machine Learning)''')
    parser.add_argument('model', metavar='MODELs', help='ML model', choices=['svm','rf','lr'])
    parser.add_argument('rep', metavar='REPETITIONs', help='Sensor configuration File.', type=int)
    parser.add_argument('kfold', metavar='K-FOLDs', help='Sensor configuration File.',type=int)
    args = parser.parse_args()
    return args

def main(args):

    print(args)

if __name__ == "__main__":
    
    args = getArguments()

    main(args)
