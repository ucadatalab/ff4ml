# -*- coding: utf-8 -*-
"""
    :mod:`main`
    ===========================================================================
    :synopsis: Multiprocessig scripts
    :author: UCADatalab - http://datalab.uca.es, NESG (Network Engineering & Security Group) - https://nesg.ugr.es
    :contact: ignacio.diaz@uca.es, roberto.magan@uca.es, rmagan@ugr.es
    :organization: University of Cádiz
    :project: ff4ml (Free Framework for Machine Learning)
    :since: 0.0.1
"""
import argparse
import multiprocessing as mp
import subprocess as sp
import shlex
import time
import os
from datetime import datetime



def getArguments():
    """
    Function to get input arguments from configuration file
    :return: args
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''ff4ml (Free Framework for Machine Learning)''')
    parser.add_argument('model', metavar='MODELs', help='ML model (svc,rf,lr)', choices=['svc','rf','lr'])
    parser.add_argument('rep', metavar='REPETITIONs', help='Repetition number (1-20).', type=int)
    parser.add_argument('kfold', metavar='K-FOLDs', help='Kfold number (1-5).',type=int)

    return parser.parse_args()


def run_external(model, rep, kfold, exec_ts):

    command = "python main.py " + model + \
              " " + str(rep) + \
              " " + str(kfold) + \
              " " + exec_ts

    os.mkdir("./results/" + exec_ts)

    print("[-] Launching {0}".format(command))
    p = sp.call(shlex.split(command))

    return p


def start_experiment(args):

    kfolds = args.kfold
    reps = args.rep
    model = args.model

    print("[+] Start experiment from model={0}, reps={1} and kfolds={2}"
          .format(model, reps, kfolds))

    cores = mp.cpu_count()
    print("[+] Available cores = {0}".format(cores))

    procs = []
    end = False

    ntasks = reps * kfolds
    print("[+] # total tasks = {0}".format(ntasks))

    while ntasks > 0:

        if len(procs) < min(cores, ntasks):
            exec_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            proc = mp.Process(target=run_external, args=(args.model,
                                                         args.rep,
                                                         args.kfold,
                                                         exec_ts))
            procs.append(proc)
            proc.start()
            print("[-] Starting task at {0}".format(exec_ts))
            print("[-] There are {0} simultaneous tasks running".format(len(procs)))

        for i, p in enumerate(procs):
            if not p.is_alive():
                del procs[i]
                ntasks = ntasks - 1

        print("[-] # of pending task = {0}".format(ntasks))
        time.sleep(2)


if __name__ == "__main__":
    args = getArguments()

    start_experiment(args)


