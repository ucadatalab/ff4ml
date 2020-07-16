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


def valid_date(s):
    """
    Check is an string has a valid date format
    :param s: string
        Date string
    :return:
    """
    try:
        return datetime.strptime(s, "%Y%m%d_%H%M%S")
    except ValueError:
        msg = "Not a valid date: '{0}'. Allowed format: %Y%m%d_%H%M%S.".format(s)
        raise argparse.ArgumentTypeError(msg)

def getArguments():
    """
    Function to get input arguments from configuration file
    :return: args
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''ff4ml (Free Framework for Machine Learning)''')
    parser.add_argument('-m','--model',
                        help='ML model (svc,rf,lr)',
                        choices=['svc', 'rf', 'lr'],
                        required=True)
    parser.add_argument('-r','--repetitions',
                        help='Total number of repetitions.',
                        type=int,
                        required=True)
    parser.add_argument('-k','--kfolds',
                        help='Total number of kfolds.',
                        type=int,
                        required=True)
    parser.add_argument('-ts','--timestamp',
                        help='Timestamp.',
                        type=valid_date)#Optional
    parser.add_argument('-cf','--configfile',
                        help='Yaml based format configuration file path.',
                        required=True)
    parser.add_argument('-ncpus', '--ncpus',
                        help='Number of simultaneously used cpus.',
                        required=True)

    return parser.parse_args()


def run_external(model, rep, kfold, exec_ts, config_file):

    command = "python main.py --model " + model + \
              " --repetition " + str(rep) + \
              " --kfold " + str(kfold) + \
              " --timestamp " + exec_ts + \
              " --configfile " + config_file

    #sp.Popen(shlex.split("mkdir -p ./results/" + exec_ts))

    print("[-] Launching {0}".format(command))
    p = sp.call(shlex.split(command))

    return p


def start_experiment(args):

    kfolds = args.kfolds
    reps = args.repetitions
    model = args.model
    ncpus = args.ncpus
    config = args.configfile

    exec_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()

    print("[+] Start experiment from model={0}, reps={1} and kfolds={2} at {3}"
          .format(model, reps, kfolds, exec_ts))

    cores = int(ncpus)
    print("[+] Available cores = {0}".format(mp.cpu_count()))
    print("[+] Selected cores = {0}".format(cores))

    procs = []

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
                                                         exec_ts,
                                                         config))

            procs.append(proc)
            proc.start()
            print("[>] Starting task (rep, kfold) = {0} at {1}"
                  .format(current_task, datetime.now().strftime("%Y%m%d_%H%M%S")))

        print("[>] There are {0} simultaneous tasks running".format(len(procs)))

        for i, p in enumerate(procs):
            if not p.is_alive():
                del procs[i]
                ntasks = ntasks - 1

        print("[>] # of pending task = {0}".format(ntasks))
        time.sleep(2)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("[+] Finished. Elapsed time {0}".format(elapsed_time))


if __name__ == "__main__":
    args = getArguments()

    start_experiment(args)


