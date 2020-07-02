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
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import classification_report
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import time
import json
import yaml

from utils import fileutils
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Models verbose
verbose = False


def getArguments():
    """
    Function to get input arguments from configuration file
    :return: args
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''ff4ml (Free Framework for Machine Learning)''')
    parser.add_argument('model', metavar='MODELs', help='ML model (svc,rf,lr)', choices=['svc', 'rf', 'lr'])
    parser.add_argument('rep', metavar='REPETITIONs', help='Repetition number (1-20).', type=int)
    parser.add_argument('kfold', metavar='K-FOLDs', help='Kfold number (1-5).', type=int)
 #   parser.add_argument('exec_ts', metavar='Timestamp', help='Timestamp.')  # Ejecución en supercomputador
    parser.add_argument('config_file', metavar='Configuration File', help='Yaml based format configuration file path.')

    return parser.parse_args()


def write_param(path_param, line, header):
    if os.path.isfile(path_param):
        file = open(path_param, "a")
    else:
        file = open(path_param, "w")
        file.write(header)
        file.write("\n")

    file.write(line)
    file.write("\n")

    file.close()
    print("[+] Line added ...")


def main(args):
    model = args.model
    rep = args.rep
    kfold = args.kfold
    yaml.warnings({'YAMLLoadWarning': False})
    # ts=args.exec_ts  # Ejecución en supercomputador
    config = fileutils.load_config(args.config_file)

    instantIni = time.time()


    print("[+] Starting task at {0} ({1},{2})".format(datetime.now(), rep, kfold))

    root_path = config['folder_paths']['root_path']
    root_path_output = config['folder_paths']['root_path_output']
    #  root_path = '../data/' # Ejecución en supercomputador
    # root_path_output = '../results/' + str(ts) + '/' # Ejecución en supercomputador

    mc_file = config['file_paths']['mc_file']
    mcfold_file = config['file_paths']['mcfold_file']
    mcvars_file = config['file_paths']['mcvars_file']

    df = pd.read_csv(root_path + mc_file, index_col=0)
    df_folds = pd.read_csv(root_path + mcfold_file)
    df_vars = pd.read_csv(root_path + mcvars_file)

    print("[+] Reading datasets ...")
    print("[-]" + mc_file + " OK")
    print("[-]" + mcfold_file + " OK")
    print("[-]" + mcvars_file + " OK")

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
    y = df['outcome.multiclass']

    # Creation of TRAINING and TEST datasets according to the number of fold.
    group = 'REP.' + str(rep)

    rows_fold = df_folds.iloc[df_folds.groupby(group).groups[kfold]].index
    No_rows_fold = df_folds[df_folds[group] != kfold][group].index

    # Getting labels from config file
    labels = config['labels']

    # Data TRAIN and LABEL
    X_train = X.drop(X.index[rows_fold])
    y_train = y.drop(y.index[rows_fold])
    y_train_bina = label_binarize(y_train, classes=labels)

    # Data TEST and LABEL
    X_test = X.drop(X.index[No_rows_fold])
    y_test = y.drop(y.index[No_rows_fold])
    y_test_bina = label_binarize(y_test, classes=labels)

    n_classes = len(labels)

    # Data normalization

    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)

    # Hyperparameters Selection

    if model == 'lr':
        title = 'LOGISTIC REGRESSION'
    elif model == 'rf':
        title = 'RANDOM FOREST'
    elif model == 'svc':
        title = 'SVC'



    print("[+] Computing hyper-parameters for the classifier: " + title + " ...")
    print("")
    if model == 'rf':
        # parameters = {'n_estimators': [2, 4, 8, 16, 32], 'max_depth': [2, 4, 8, 16]}
        # parameters = {'n_estimators': 500, 'max_features': [2, 4, 8, 16]}
        parameters = {'n_estimators': Integer(config['models_hyper']['rf']['parameters']['n_estimators']['low_val'],config['models_hyper']['rf']['parameters']['n_estimators']['max_val']), 'max_features': Integer(config['models_hyper']['rf']['parameters']['max_features']['low_val'],config['models_hyper']['rf']['parameters']['max_features']['max_val'])}
        # parameters = {'n_estimators': Integer(500,600), 'max_features': Integer(2,16)}
        # model_grid = RandomForestClassifier(random_state=0, n_jobs=2)
        model_grid = RandomForestClassifier(random_state = config['model_grid']['rf']['random_state'],n_jobs = config['model_grid']['rf']['n_jobs'])

    elif model == 'svc':
        # parameters = {'gamma': [2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1], 'C': [0.1, 1, 10, 100]}
        parameters = {
            'C': Real(config['models_hyper']['svc']['parameters']['C']['low_val'],config['models_hyper']['svc']['parameters']['C']['max_val'], prior='log-uniform'),
            'gamma': Real(2e-3, 2, prior='log-uniform')
            # 'models_hyper']['svc']['parameters']['gamma']['low_val'], config['models_hyper']['svc']['parameters']['gamma']['max_val'], prior='log-uniform')
        }

        # model_grid = SVC(random_state=0, kernel='rbf')
        model_grid = SVC(random_state= config['model_grid']['svc']['random_state'], kernel = config['model_grid']['svc']['kernel'])

    if model != 'lr':
        clf = BayesSearchCV(model_grid, parameters,
                            n_iter= config['hyperparameters']['n_iter'], n_jobs= config['hyperparameters']['n_jobs'], cv = config['hyperparameters']['cv'], n_points= config['hyperparameters']['n_points'])
                            # n_iter = 30, n_jobs = 3, cv = 5, n_points = 8)
        # clf = GridSearchCV(model_grid, parameters, cv=5, verbose=verbose)
        clf.fit(X_train_scaled, y_train)
        print("")
        print("[+] The best parameters for " + "Rep.: " + str(rep) + " and Kfold: " + str(kfold) + " are:  [+]")
        print(str(clf.best_params_))
        print("")
        bp = clf.best_params_

        # Save selected parameters to .json

        path_param_output_json_bp = root_path_output + "PARAMETERS_" + model + "_" + str(rep) + "_" + str(
            kfold) + "_" + "output" + ".json"
        with open(path_param_output_json_bp, "w") as fi:
            json.dump(bp, fi)

        print("---BEST PARAMETERS SAVED ---")

    # PARAMETERS SELECTED

    print("[+] PARAMETERS SELECTED MODEL " + title + " [+]")
    print("")
    if model == 'rf':
        # md = int(bp.get('max_depth'))
        mf = int(bp.get('max_features'))
        ne = int(bp.get('n_estimators'))
        print("Max_Features: ", mf)
        print("n_estimators: ", ne)
        # nit = int(bp.get('n_estimators'))
        # print("N_Estimators: ", nit)
        # tmodel = RandomForestClassifier(min_samples_split=2, min_samples_leaf=2, max_depth=md, random_state=0, n_estimators=nit, verbose=verbose)
        tmodel = clf
    elif model == 'lr':
        print("Solver: lbfgs")
        print("Multi_class: auto")
        print("Penalty: none")
        #tmodel = LogisticRegression(random_state=0, penalty='none', multi_class='auto',
                                   # solver='lbfgs', verbose=verbose)
        print("TMODEL")
        tmodel = LogisticRegression(random_state= config['model_grid']['lr']['random_state'], penalty = config['model_grid']['lr']['penalty'], multi_class = config['model_grid']['lr']['multi_class'],
                                    solver= config['model_grid']['lr']['solver'], verbose= config['model_grid']['lr']['verbose'])

    elif model == 'svc':
        cs = float(bp.get('C'))
        print("cs: ", cs)
        ga = float(bp.get('gamma'))
        print("ga: ", ga)
        tmodel = clf

    # Training models
    print("[+] TRAINING MODELS " + "[+]")
    print("")
    print("[+] Model Training: " + title + "\n")

    tmodeldef = OneVsRestClassifier(tmodel)

    tmodeldef.fit(X_train_scaled, y_train_bina)
    print("")
    print("[+] MODEL PREDICTING " + model + "\n")
    predictions_test = tmodeldef.predict(X_test_scaled)
    predictions_train = tmodeldef.predict(X_train_scaled)
    print("")

    print("[+] CLASSIFICATION REPORT TEST " + model + "\n")
    clasif_test = classification_report(y_test_bina, predictions_test, output_dict=True, target_names=labels)
    print(classification_report(y_test_bina, predictions_test, target_names=labels))
    print("")
    print("[+] CLASSIFICATION REPORT TRAIN " + model + "\n")
    clasif_train = classification_report(y_train_bina, predictions_train, output_dict=True, target_names=labels)
    print(classification_report(y_train_bina, predictions_train, target_names=labels))
    print("")

    # Compute ROC area for each class TEST
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(n_classes):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test_bina[:, i], predictions_test[:, i])
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Compute per class ROCs and AUCs
    supports_sum_test = 0
    auc_partial_test = 0

    for label in config['labels']:
        supports_sum_test = supports_sum_test + (clasif_test[label]['support'])
        auc_partial_test = auc_partial_test + ((clasif_test[label]['support']) * roc_auc_test[i])
    auc_w_test = auc_partial_test / supports_sum_test

    print("SUM SUPPORTS TEST: ", supports_sum_test)
    print("AUC_W TEST: ", auc_w_test)
    print("")

    # Compute ROC area for each class TRAIN
    fpr_train = dict()
    tpr_train = dict()
    roc_auc_train = dict()

    for i in range(n_classes):
        fpr_train[i], tpr_train[i], _ = roc_curve(y_train_bina[:, i], predictions_train[:, i])
        roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])

    # Compute per class ROCs and AUCs
    supports_sum_train = 0
    auc_partial_train = 0

    for label in config['labels']:
        supports_sum_train = supports_sum_train + (clasif_train[label]['support'])
        auc_partial_train = auc_partial_train + ((clasif_train[label]['support']) * roc_auc_train[i])
    auc_w_train = auc_partial_train / supports_sum_train

    print("SUM SUPPORTS TRAIN: ", supports_sum_train)
    print("AUC_W TRAIN: ", auc_w_train)
    print("")

    # Send data to .csv

    instantFinal = time.time()
    # elapsed time in seconds
    elapsedtime = instantFinal - instantIni
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


    '''
# Automatically building the header according to the labels.
    # head of header
    header = "Rep." + \
             "," + "Kfold" + \
             "," + "Num. Vars."

    # body of header
    for label in labels:
        header = header + \
                 "," + "Precision-" + label + \
                 "," + "Recall-" + label + \
                 "," + "F1_score_" + label + \
                 "," + "Num. Obs. " + label + \
                 "," + "AUC_" + label

    # tail of header
    header = header + \
             "," + "Precision-w" + \
             "," + "Recall-w" + \
             "," + "F1_score_w" + \
             "," + "Total Obs." + \
             "," + "AUC_w" + \
             "," + "Time"

    # TEST RESULTS
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

    line_test.append(auc_w_test)
    line_test.append(elapsedtime)

    test_df = pd.DataFrame([1, 2], names=header)
    test_df.to_csv('path.csv')
    '''

    header = "Rep." + \
             "," + "Kfold" + \
             "," + "Num. Vars." + \
             "," + "Precision-Background" + \
             "," + "Recall-Background" + \
             "," + "F1_score_Background" + \
             "," + "Num. Obs. Background" + \
             "," + "AUC_Background" + \
             "," + "Precision-DoS" + \
             "," + "Recall-DoS" + \
             "," + "F1_score_DoS" + \
             "," + "Num. Obs. Dos" + \
             "," + "AUC_DoS" + \
             "," + "Precision-Botnet" + \
             "," + "Recall-Botnet" + \
             "," + "F1_score_Botnet" + \
             "," + "Num. Obs. Botnet" + \
             "," + "AUC_Botnet" + \
             "," + "Precision-Scan" + \
             "," + "Recall-Scan" + \
             "," + "F1_score_Scan" + \
             "," + "Num. Obs. Scan" + \
             "," + "AUC_Scan" + \
             "," + "Precision-SSHscan" + \
             "," + "Recall-SSHscan" + \
             "," + "F1_score_SSHscan" + \
             "," + "Num. Obs. SSHscan" + \
             "," + "AUC_SSHscan" + \
             "," + "Precision-UDPscan" + \
             "," + "Recall-UDPscan" + \
             "," + "F1_score_UDPscan" + \
             "," + "Num. Obs. UDPscan" + \
             "," + "AUC_UDPscan" + \
             "," + "Precision-Spam" + \
             "," + "Recall-Spam" + \
             "," + "F1_score_Spam" + \
             "," + "Num. Obs. Spam" + \
             "," + "AUC_Spam" + \
             "," + "Precision-w" + \
             "," + "Recall-w" + \
             "," + "F1_score_w" + \
             "," + "Total Obs." + \
             "," + "AUC_w" + \
             "," + "Time"

    line_test = str(rep) + \
                ',' + str(kfold) + \
                ',' + str(len(f)) + \
                ',' + str(clasif_test['background']['precision']) + \
                ',' + str(clasif_test['background']['recall']) + \
                ',' + str(clasif_test['background']['f1-score']) + \
                ',' + str(clasif_test['background']['support']) + \
                ',' + str(roc_auc_test[0]) + \
                ',' + str(clasif_test['dos']['precision']) + \
                ',' + str(clasif_test['dos']['recall']) + \
                ',' + str(clasif_test['dos']['f1-score']) + \
                ',' + str(clasif_test['dos']['support']) + \
                ',' + str(roc_auc_test[1]) + \
                ',' + str(clasif_test['nerisbotnet']['precision']) + \
                ',' + str(clasif_test['nerisbotnet']['recall']) + \
                ',' + str(clasif_test['nerisbotnet']['f1-score']) + \
                ',' + str(clasif_test['nerisbotnet']['support']) + \
                ',' + str(roc_auc_test[2]) + \
                ',' + str(clasif_test['scan']['precision']) + \
                ',' + str(clasif_test['scan']['recall']) + \
                ',' + str(clasif_test['scan']['f1-score']) + \
                ',' + str(clasif_test['scan']['support']) + \
                ',' + str(roc_auc_test[3]) + \
                ',' + str(clasif_test['sshscan']['precision']) + \
                ',' + str(clasif_test['sshscan']['recall']) + \
                ',' + str(clasif_test['sshscan']['f1-score']) + \
                ',' + str(clasif_test['sshscan']['support']) + \
                ',' + str(roc_auc_test[4]) + \
                ',' + str(clasif_test['udpscan']['precision']) + \
                ',' + str(clasif_test['udpscan']['recall']) + \
                ',' + str(clasif_test['udpscan']['f1-score']) + \
                ',' + str(clasif_test['udpscan']['support']) + \
                ',' + str(roc_auc_test[5]) + \
                ',' + str(clasif_test['spam']['precision']) + \
                ',' + str(clasif_test['spam']['recall']) + \
                ',' + str(clasif_test['spam']['f1-score']) + \
                ',' + str(clasif_test['spam']['support']) + \
                ',' + str(roc_auc_test[6]) + \
                ',' + str(clasif_test['weighted avg']['precision']) + \
                ',' + str(clasif_test['weighted avg']['recall']) + \
                ',' + str(clasif_test['weighted avg']['f1-score']) + \
                ',' + str(clasif_test['weighted avg']['support']) + \
                ',' + str(auc_w_test) + \
                ',' + str(elapsedtime)

    line_train = str(rep) + \
                 ',' + str(kfold) + \
                 ',' + str(len(f)) + \
                 ',' + str(clasif_train['background']['precision']) + \
                 ',' + str(clasif_train['background']['recall']) + \
                 ',' + str(clasif_train['background']['f1-score']) + \
                 ',' + str(clasif_train['background']['support']) + \
                 ',' + str(roc_auc_train[0]) + \
                 ',' + str(clasif_train['dos']['precision']) + \
                 ',' + str(clasif_train['dos']['recall']) + \
                 ',' + str(clasif_train['dos']['f1-score']) + \
                 ',' + str(clasif_train['dos']['support']) + \
                 ',' + str(roc_auc_train[1]) + \
                 ',' + str(clasif_train['nerisbotnet']['precision']) + \
                 ',' + str(clasif_train['nerisbotnet']['recall']) + \
                 ',' + str(clasif_train['nerisbotnet']['f1-score']) + \
                 ',' + str(clasif_train['nerisbotnet']['support']) + \
                 ',' + str(roc_auc_train[2]) + \
                 ',' + str(clasif_train['scan']['precision']) + \
                 ',' + str(clasif_train['scan']['recall']) + \
                 ',' + str(clasif_train['scan']['f1-score']) + \
                 ',' + str(clasif_train['scan']['support']) + \
                 ',' + str(roc_auc_train[3]) + \
                 ',' + str(clasif_train['sshscan']['precision']) + \
                 ',' + str(clasif_train['sshscan']['recall']) + \
                 ',' + str(clasif_train['sshscan']['f1-score']) + \
                 ',' + str(clasif_train['sshscan']['support']) + \
                 ',' + str(roc_auc_train[4]) + \
                 ',' + str(clasif_train['udpscan']['precision']) + \
                 ',' + str(clasif_train['udpscan']['recall']) + \
                 ',' + str(clasif_train['udpscan']['f1-score']) + \
                 ',' + str(clasif_train['udpscan']['support']) + \
                 ',' + str(roc_auc_train[5]) + \
                 ',' + str(clasif_train['spam']['precision']) + \
                 ',' + str(clasif_train['spam']['recall']) + \
                 ',' + str(clasif_train['spam']['f1-score']) + \
                 ',' + str(clasif_train['spam']['support']) + \
                 ',' + str(roc_auc_train[6]) + \
                 ',' + str(clasif_train['weighted avg']['precision']) + \
                 ',' + str(clasif_train['weighted avg']['recall']) + \
                 ',' + str(clasif_train['weighted avg']['f1-score']) + \
                 ',' + str(clasif_train['weighted avg']['support']) + \
                 ',' + str(auc_w_train) + \
                 ',' + str(elapsedtime)

    write_param(path_param_output_test, line_test, header)
    write_param(path_param_output_train, line_train, header)

    # Send data to .json

    with open(path_param_output_json_fpr_test, "w") as fpr_dict:
        for name, value in fpr_test.items():
            fpr_dict.write("%s %s\n" % (labels[int(name)], value))
        print("---FPR TEST WRITTEN---")

    with open(path_param_output_json_tpr_test, "w") as tpr_dict:
        for name, value in tpr_test.items():
            tpr_dict.write("%s %s\n" % (labels[int(name)], value))
        print("---TPR TEST WRITTEN---")

    with open(path_param_output_json_fpr_train, "w") as fpr_dict:
        for name, value in fpr_train.items():
            fpr_dict.write("%s %s\n" % (labels[int(name)], value))
        print("---FPR TRAIN WRITTEN---")

    with open(path_param_output_json_tpr_train, "w") as tpr_dict:
        for name, value in tpr_train.items():
            tpr_dict.write("%s %s\n" % (labels[int(name)], value))
        print("---TPR TRAIN WRITTEN---")

    print("------------------")
    print(" [+] Time Stamp: ---" +
          " REP: ---" + str(rep) +
          "---" + " Kfold: " +
          "---" + str(kfold) +
          "--- Model: ---" + title +
          "---" + " FINISHED! [+]")
    print("------------------")
    print("Elapsed time (s): ", elapsedtime)

    print("[+] Finishing task at {0} ({1},{2})".format(datetime.now(), rep, kfold))


if __name__ == "__main__":
    args = getArguments()

    main(args)
