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
import json

# Models verbose
verbose = False

# Labels mapping
classes_map = {'background': 0,
               'dos': 1,
               'nerisbotnet': 2,
               'scan': 3,
               'sshscan': 4,
               'udpscan': 5,
               'spam': 6}

# Labels
labels = list(classes_map.keys())

# Labels' IDs
ids = list(classes_map.values())


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
    parser.add_argument('exec_ts', metavar='Timestamp', help='Timestamp.') # Ejecución en supercomputador

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
    model=args.model
    rep=args.rep
    kfold=args.kfold
    ts=args.exec_ts  # Ejecución en supercomputador

    instantIni = datetime.now()

    root_path = './data/'
    root_path_output = './results/' + str(ts) + '/'
    
    mc_file = 'ugr16_multiclass.csv'
    mcfold_file = 'ugr16_multiclass_folds.csv'
    mcvars_file = 'ugr16_multiclass_folds_selecvars.csv'

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
    y = df['outcome.multiclass'].map(classes_map)

    # Creation of TRAINING and TEST datasets according to the number of fold.
    group = 'REP.' + str(rep)

    rows_fold = df_folds.iloc[df_folds.groupby(group).groups[kfold]].index
    No_rows_fold = df_folds[df_folds[group] != kfold][group].index

    # Data TRAIN and LABEL
    X_train = X.drop(X.index[rows_fold])
    y_train = y.drop(y.index[rows_fold])
    y_train_bina = label_binarize(y_train, classes=ids)

    # Data TEST and LABEL
    X_test = X.drop(X.index[No_rows_fold])
    y_test = y.drop(y.index[No_rows_fold])
    y_test_bina = label_binarize(y_test, classes=ids)

    n_classes = y_train_bina.shape[1]

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

    print("[+] Calculating hyper-parameters for the classifier: " + title + " ...")
    print("")
    if model == 'rf':
        parameters = {'n_estimators': [2, 4, 8, 16, 32], 'max_depth': [2, 4, 8, 16]}
        model_grid = RandomForestClassifier(random_state=0, min_samples_split=2, min_samples_leaf=2)
    elif model == 'svc':
        parameters = {'gamma': [2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1], 'C': [0.1, 1, 10, 100]}

        model_grid = SVC(random_state=0, kernel='rbf')

    if model != 'lr':
        clf = GridSearchCV(model_grid, parameters, cv=5, verbose=verbose)
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

        print("---BEST PARAMETERS WRITED---")

    # PARAMETERS SELECTED

    print("[+] PARAMETERS SELECTED MODEL " + title + " [+]")
    print("")
    if model == 'rf':
        md = int(bp.get('max_depth'))
        print("Max_Depth: ", md)
        nit = int(bp.get('n_estimators'))
        print("N_Estimators: ", nit)
        tmodel = RandomForestClassifier(min_samples_split=2, min_samples_leaf=2, max_depth=md,
                                        random_state=0, n_estimators=nit, verbose=verbose)
    elif model == 'lr':
        print("Solver: lbfgs")
        print("Multi_class: auto")
        print("Penalty: none")
        tmodel = LogisticRegression(random_state=0, penalty='none', multi_class='auto',
                                    solver='lbfgs', verbose=verbose)
    elif model == 'svc':
        cs = int(bp.get('C'))
        print("cs: ", cs)
        ga = float(bp.get('gamma'))
        print("ga: ", ga)
        tmodel = SVC(random_state=0, kernel='rbf', gamma=ga, C=cs, verbose=verbose)

    # Training models
    print("[+] TRAINING MODELS " + "[+]")
    print("")
    print("[+] Model Training: " + title + "\n")

    tmodeldef = OneVsRestClassifier(tmodel)

    tmodeldef.fit(X_train_scaled, y_train_bina)
    print("")
    print("[+] MODEL PREDICTING " + model + "\n")
    predictions = tmodeldef.predict(X_test_scaled)
    print("")

    print("[+] CLASSIFICATION REPORT " + model + "\n")
    h = classification_report(y_test_bina, predictions, output_dict=True, target_names=labels)
    print(classification_report(y_test_bina, predictions, target_names=labels))
    print("")

    # Compute ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bina[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute per class ROCs and AUCs
    supports_sum = 0
    auc_partial = 0

    for i in range(len(labels)):
        supports_sum = supports_sum + (h[labels[i]]['support'])
        auc_partial = auc_partial + ((h[labels[i]]['support']) * roc_auc[i])
    auc_w = auc_partial / supports_sum

    print("SUM SUPPORTS: ", supports_sum)
    print("AUC_W: ", auc_w)
    print("")

    # Send data to .csv

    instantFinal = datetime.now()
    time = instantFinal - instantIni
    path_param_output = root_path_output + model + "_" + str(rep) + "_" + str(kfold) + "_" + "output" + ".csv"
    path_param_output_json_fpr = root_path_output + "FPR_" + model + "_" + str(rep) + "_" + str(kfold) + "_" + "output" + ".json"
    path_param_output_json_tpr = root_path_output + "TPR_" + model + "_" + str(rep) + "_" + str(kfold) + "_" + "output" + ".json"

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

    line = str(rep) + \
           ',' + str(kfold) + \
           ',' + str(len(f)) + \
           ',' + str(h['background']['precision']) + \
           ',' + str(h['background']['recall']) + \
           ',' + str(h['background']['f1-score']) + \
           ',' + str(h['background']['support']) + \
           ',' + str(roc_auc[0]) + \
           ',' + str(h['dos']['precision']) + \
           ',' + str(h['dos']['recall']) + \
           ',' + str(h['dos']['f1-score']) + \
           ',' + str(h['dos']['support']) + \
           ',' + str(roc_auc[1]) + \
           ',' + str(h['nerisbotnet']['precision']) + \
           ',' + str(h['nerisbotnet']['recall']) + \
           ',' + str(h['nerisbotnet']['f1-score']) + \
           ',' + str(h['nerisbotnet']['support']) + \
           ',' + str(roc_auc[2]) + \
           ',' + str(h['scan']['precision']) + \
           ',' + str(h['scan']['recall']) + \
           ',' + str(h['scan']['f1-score']) + \
           ',' + str(h['scan']['support']) + \
           ',' + str(roc_auc[3]) + \
           ',' + str(h['sshscan']['precision']) + \
           ',' + str(h['sshscan']['recall']) + \
           ',' + str(h['sshscan']['f1-score']) + \
           ',' + str(h['sshscan']['support']) + \
           ',' + str(roc_auc[4]) + \
           ',' + str(h['udpscan']['precision']) + \
           ',' + str(h['udpscan']['recall']) + \
           ',' + str(h['udpscan']['f1-score']) + \
           ',' + str(h['udpscan']['support']) + \
           ',' + str(roc_auc[5]) + \
           ',' + str(h['spam']['precision']) + \
           ',' + str(h['spam']['recall']) + \
           ',' + str(h['spam']['f1-score']) + \
           ',' + str(h['spam']['support']) + \
           ',' + str(roc_auc[6]) + \
           ',' + str(h['weighted avg']['precision']) + \
           ',' + str(h['weighted avg']['recall']) + \
           ',' + str(h['weighted avg']['f1-score']) + \
           ',' + str(h['weighted avg']['support']) + \
           ',' + str(auc_w) + \
           ',' + str(time)

    write_param(path_param_output, line, header)

    # Send data to .json
    with open(path_param_output_json_fpr, "w") as fpr_dict:
        for name, value in fpr.items():
            fpr_dict.write("%s %s\n" % (labels[int(name)], value))
        print("---FPR WRITTEN---")

    with open(path_param_output_json_tpr, "w") as tpr_dict:
        for name, value in tpr.items():
            tpr_dict.write("%s %s\n" % (labels[int(name)], value))
        print("---TPR WRITTEN---")

    print("------------------")
    print(" [+] Time Stamp: ---" +
          " REP: ---" + str(rep) +
          "---" + " Kfold: " +
          "---" + str(kfold) +
          "--- Model: ---" + title +
          "---" + " FINISHED! [+]")
    print("------------------")
    print("Elapsed time: ", time)


if __name__ == "__main__":
    args = getArguments()

    main(args)
