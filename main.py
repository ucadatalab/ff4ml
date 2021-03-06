# -*- coding: utf-8 -*-
"""
    :mod:`main`
    ===========================================================================
    :synopsis: Main class
    :author: UCADatalab - http://datalab.uca.es, NESG (Network Engineering & Security Group) - https://nesg.ugr.es
    :contact: ignacio.diaz@uca.es, roberto.magan@uca.es, rmagan@ugr.es
    :organization: University of Cádiz
    :project: ff4ml (Free Framework for Machine Learning)
    :since: 0.0.1
"""

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import time
from datetime import datetime
import os

from utils import fileutils
from skopt import BayesSearchCV
from skopt.space import Real, Integer


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
    parser.add_argument('-r','--repetition',
                        help='Repetition number (1-20).',
                        type=int,
                        required=True)
    parser.add_argument('-k','--kfold',
                        help='Kfold number (1-5).',
                        type=int,
                        required=True)
    parser.add_argument('-ts','--timestamp',
                        help='Timestamp.',
                        type=valid_date)#Optional
    parser.add_argument('-cf','--configfile',
                        help='Yaml based format configuration file path.',
                        required=True)

    return parser.parse_args()


def main(args):
    model = args.model
    rep = args.repetition
    kfold = args.kfold
    config = fileutils.load_config(args.configfile)

    # When executing this from the cluster, the timestamp has been set up before
    if args.timestamp:
        ts = datetime.strftime(args.timestamp,"%Y%m%d_%H%M%S")
    else:
        ts = datetime.strftime(datetime.today(),"%Y%m%d_%H%M%S")

    # Data root path
    root_path = config['folder_paths']['root_path']
    # Result root path
    root_path_output = config['folder_paths']['root_path_output'] \
                       + config['dataset']['name'] + os.path.sep + ts + os.path.sep

    try:
        # Create result paths
        if not os.path.exists(root_path_output):
            os.makedirs(root_path_output)
    except OSError as oe:
        print("ERROR: Sensor results directory cannot be created: %s", oe)
        print("Exiting ...")
        exit(1)

    instantIni = time.time()

    print("[+] Starting task at {0} - ({1},{2}) - model: {3}".format(datetime.now(), rep, kfold, model))
    print("[+] Task configuration:")
    print(fileutils.print_config(config))
   
    mc_file = config['file_paths']['mc_file']
    mcfold_file = config['file_paths']['mcfold_file']
    mcvars_file = config['file_paths']['mcvars_file']

    df = pd.read_csv(root_path + mc_file)
    df_folds = pd.read_csv(root_path + mcfold_file)
    df_vars = pd.read_csv(root_path + mcvars_file)

    print("[+] Reading datasets ...")

# Feature selection

    d = (df_vars.groupby('repeticion').groups[rep]) & (df_vars.groupby('caja.de.test').groups[kfold])

    size = df_vars.shape[1]
    f = ['']

    for i in range(1, size):
        if int(df_vars.iloc[d, i]) == 1:
            f.append(df_vars.iloc[d, i].name)

    f.remove("")
    if kfold == 1:
        f.remove("caja.de.test")

# Data separation and label

    X = df[f]
    y = df['outcome']

# Creation of TRAINING and TEST datasets according to the number of fold

    print("[+] Getting the repetition and fold ...")

    group = 'REP.' + str(rep)
    rows_fold = df_folds.iloc[df_folds.groupby(group).groups[kfold]].index
    No_rows_fold = df_folds[df_folds[group] != kfold][group].index

# Getting labels from config file

    labels = config['labels']

    print("[+] Building train and test parts ...")

# Data TRAIN and LABEL

    X_train = X.drop(X.index[rows_fold])
    y_train = y.drop(y.index[rows_fold])
    y_train_bina = label_binarize(y_train, classes=labels)


# Data TEST and LABEL

    X_test = X.drop(X.index[No_rows_fold])
    y_test = y.drop(y.index[No_rows_fold])
    y_test_bina = label_binarize(y_test, classes=labels)

# Data normalization

    print("[+] Scaling data ...")

    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)

# Hyperparameters Selection

    if model == 'rf':
        parameters = {'n_estimators': Integer(config['models']['rf']['parameters']['n_estimators'][0],
                                              config['models']['rf']['parameters']['n_estimators'][1]),
                      'max_features': Integer(config['models']['rf']['parameters']['max_features'][0],
                                              config['models']['rf']['parameters']['max_features'][1])}
        model_grid = RandomForestClassifier(random_state = config['models']['rf']['general']['random_state'],
                                            n_jobs = config['models']['rf']['general']['n_jobs'])

    elif model == 'svc':
        parameters = {
            'C': Real(config['models']['svc']['hyperparameters']['C'][0],
                      config['models']['svc']['hyperparameters']['C'][1],
                      prior='log-uniform')
        }

        # Adding gamma when SVC kernel is rbf
        if config['models']['svc']['hyperparameters']['kernel'] == 'rbf':
            parameters['gamma'] = Real(config['models']['svc']['hyperparameters']['gamma'][0],
                                       config['models']['svc']['hyperparameters']['gamma'][1],
                                       prior='log-uniform')


        model_grid = SVC(random_state= config['models']['svc']['general']['random_state'],
                         kernel = config['models']['svc']['hyperparameters']['kernel'])

    if model != 'lr':
        print("[+] Computing hyper-parameters ...")
        print("[>] Hyper-parameters: {0}".format(parameters))
        clf = BayesSearchCV(model_grid, parameters,
                            n_iter= config['hyper_bayesian']['n_iter'],
                            n_jobs= config['hyper_bayesian']['n_jobs'],
                            cv = config['hyper_bayesian']['cv'],
                            n_points = config['hyper_bayesian']['n_points'],
                            random_state = config['hyper_bayesian']['random_state'])

        clf.fit(X_train_scaled, y_train)
        print("")
        print("[>] The best parameters for " + "Rep.: " + str(rep) + " and Kfold: " + str(kfold) + " are:  [+]")
        print(str(clf.best_params_))
        print("")
        bp = clf.best_params_
        # Built model with the selected parameters
        tmodel = clf

        # Saving selected parameters to .json
        path_param_output_json_bp = root_path_output + "PARAMETERS_" + model + "_" + str(rep) + "_" + str(
            kfold) + "_" + "output" + ".json"
        fileutils.params_to_json(bp, path_param_output_json_bp)
    else:
        tmodel = LogisticRegression(random_state=config['models']['lr']['random_state'],
                                    penalty=config['models']['lr']['penalty'],
                                    multi_class=config['models']['lr']['multi_class'],
                                    solver=config['models']['lr']['solver'],
                                    verbose=config['models']['lr']['verbose'])

# Training models

    print("[+] Training ...")

    # Each class is modeled separately.
    tmodeldef = OneVsRestClassifier(tmodel)
    tmodeldef.fit(X_train_scaled, y_train_bina)
    predictions_train = tmodeldef.predict(X_train_scaled)

    print("[>] Train performance ...")
    clasif_train = classification_report(y_train_bina, predictions_train, output_dict=True, target_names=labels)
    print(classification_report(y_train_bina, predictions_train, target_names=labels))

    # Compute ROC and AUCs per class

    fpr_train = dict()
    tpr_train = dict()
    roc_auc_train = dict()

    for i, label in enumerate(labels):
        fpr_train[i], tpr_train[i], _ = roc_curve(y_train_bina[:, i], predictions_train[:, i])
        roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])

    # weighted average

    supports_sum_train = 0
    auc_partial_train = 0

    for i, label in enumerate(labels):
        supports_sum_train = supports_sum_train + (clasif_train[label]['support'])
        auc_partial_train = auc_partial_train + ((clasif_train[label]['support']) * roc_auc_train[i])
    auc_w_train = auc_partial_train / supports_sum_train

    print("[>] Train total supports {0}".format(supports_sum_train))
    print("[>] Train AUC weighted average {0}".format(auc_w_train))

    print("[+] Testing ... ")
    predictions_test = tmodeldef.predict(X_test_scaled)

    print("[>] Test performance ...")
    clasif_test = classification_report(y_test_bina, predictions_test, output_dict=True, target_names=labels)
    print(classification_report(y_test_bina, predictions_test, target_names=labels))

    # Compute ROC and AUCs per class

    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i, label in enumerate(labels):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test_bina[:, i], predictions_test[:, i])
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])
    print(labels)
    print(roc_auc_test)

# weighted average

    supports_sum_test = 0
    auc_partial_test = 0

    for i, label in enumerate(labels):
        supports_sum_test = supports_sum_test + (clasif_test[label]['support'])
        auc_partial_test = auc_partial_test + ((clasif_test[label]['support']) * roc_auc_test[i])
    auc_w_test = auc_partial_test / supports_sum_test

    print("[>] Test total supports {0}".format(supports_sum_test))
    print("[>] Test AUC weighted average {0}".format(auc_w_test))

# Elapsed time in seconds

    instantFinal = time.time()
    elapsedtime = instantFinal - instantIni

    print("[+] Saving results ...")

# Output paths

    path_param_output_test = root_path_output + model + "_" + str(rep) + "_" + str(kfold) + "_" + "output_test" + ".csv"
    path_param_output_train = root_path_output + model + "_" + str(rep) + "_" + str(
        kfold) + "_" + "output_train" + ".csv"

    path_param_output_json_fpr_test = root_path_output + "FPR_" + model + "_" + str(rep) + "_" + str(
        kfold) + "_" + "output_test" + ".json"
    path_param_output_json_tpr_test = root_path_output + "TPR_" + model + "_" + str(rep) + "_" + str(
        kfold) + "_" + "output_test" + ".json"

    path_param_output_json_fpr_train = root_path_output + "FPR_" + model + "_" + str(rep) + "_" + str(
        kfold) + "_" + "output_train" + ".json"
    path_param_output_json_tpr_train = root_path_output + "TPR_" + model + "_" + str(rep) + "_" + str(
        kfold) + "_" + "output_train" + ".json"

# Automatically building header and rows according to the labels.

    # Head of header

    h = []
    h.append("Rep")
    h.append("Kfold")
    h.append("Num_Vars_")

    # Body of header

    for label in labels:
        l = "Precision_" + label
        h.append(l)
        l = "Recall_" + label
        h.append(l)
        l = "F1_score_" + label
        h.append(l)
        l = "Num_Obs_" + label
        h.append (l)
        l = "AUC_" + label
        h.append(l)

    # Tail of header

    h.append("Precision_w")
    h.append("Recall_w")
    h.append("F1_score_w")
    h.append ("Total_Obs")
    h.append("AUC_w")
    h.append("Time")

    # Train results to .csv

    line_train = []
    line_train.append(rep)
    line_train.append(kfold)
    line_train.append(len(f))  # Number of selected variables

    for i, label in enumerate(labels):
        line_train.append(clasif_train[label]['precision'])
        line_train.append(clasif_train[label]['recall'])
        line_train.append(clasif_train[label]['f1-score'])
        line_train.append(clasif_train[label]['support'])
        line_train.append(roc_auc_train[i])

    line_train.append(clasif_train['weighted avg']['precision'])
    line_train.append(clasif_train['weighted avg']['recall'])
    line_train.append(clasif_train['weighted avg']['f1-score'])
    line_train.append(clasif_train['weighted avg']['support'])
    line_train.append(auc_w_train)
    line_train.append(elapsedtime)

    print("[>] Saving train results ...")

    data_train = pd.DataFrame(line_train, h)
    data_train = data_train.T
    data_train.to_csv(path_param_output_train, index=False)

    # Test results to .csv

    line_test = []
    line_test.append(rep)
    line_test.append(kfold)
    line_test.append(len(f)) # Number of selected variables

    for i,label in enumerate(labels):
        line_test.append(clasif_test[label]['precision'])
        line_test.append(clasif_test[label]['recall'])
        line_test.append(clasif_test[label]['f1-score'])
        line_test.append(clasif_test[label]['support'])
        line_test.append(roc_auc_test[i])

    line_test.append(clasif_test['weighted avg']['precision'])
    line_test.append(clasif_test['weighted avg']['recall'])
    line_test.append(clasif_test['weighted avg']['f1-score'])
    line_test.append(clasif_test['weighted avg']['support'])
    line_test.append(auc_w_test)
    line_test.append(elapsedtime)

    print("[>] Saving test results ...")

    data_test = pd.DataFrame(line_test, h)
    data_test = data_test.T
    data_test.to_csv(path_param_output_test,index=False)


# Send data to .json

    print("[>] Saving FPR and TPR results from train and test ...")

    with open(path_param_output_json_fpr_test, "w") as fpr_dict:
        for name, value in fpr_test.items():
            fpr_dict.write("%s %s\n" % (labels[int(name)], value))

    with open(path_param_output_json_tpr_test, "w") as tpr_dict:
        for name, value in tpr_test.items():
            tpr_dict.write("%s %s\n" % (labels[int(name)], value))

    with open(path_param_output_json_fpr_train, "w") as fpr_dict:
        for name, value in fpr_train.items():
            fpr_dict.write("%s %s\n" % (labels[int(name)], value))

    with open(path_param_output_json_tpr_train, "w") as tpr_dict:
        for name, value in tpr_train.items():
            tpr_dict.write("%s %s\n" % (labels[int(name)], value))

    print("[+] Finishing task at {0} - ({1},{2}) - model: {3}".format(datetime.now(), rep, kfold, model))
    print("[+] Elapsed time (s): ", elapsedtime)


if __name__ == "__main__":
    args = getArguments()

    main(args)
