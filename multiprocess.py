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

    sp.Popen(shlex.split("mkdir -p ./results/" + exec_ts))

    print("[-] Launching {0}".format(command))
    p = sp.call(shlex.split(command))

    return p


def start_experiment(args):

    kfolds = args.kfold
    reps = args.rep
    model = args.model

    exec_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("[+] Start experiment from model={0}, reps={1} and kfolds={2} at {3}"
          .format(model, reps, kfolds, exec_ts))

    cores = mp.cpu_count()
    print("[+] Available cores = {0}".format(cores))

    procs = []
    end = False

    # Creating list of tasks
    task_queue = [(rep, kfold) for rep in range(1, reps + 1, 1) for kfold in range(1, kfolds + 1, 1)]
    ntasks = len(task_queue)
    print("[+] # total tasks = {0}".format(ntasks))

    while ntasks > 0:

        if len(procs) < min(cores, ntasks):
            # Getting the task
            current_task = task_queue[0]
            del task_queue[0]
            proc = mp.Process(target=run_external, args=(args.model,
                                                         current_task[0], # Repetition
                                                         current_task[1], # Fold
                                                         exec_ts))
            procs.append(proc)
            proc.start()
            print("[-] Starting task (rep, kfold) = {0} at {1}"
                  .format(current_task, datetime.now().strftime("%Y%m%d_%H%M%S")))

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


