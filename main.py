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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import classification_report
from datetime import datetime
import sys
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

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

def write_param(path_param, line, header):
    if os.path.isfile(path_param):
        f = open(path_param, "a")
    else:
        f = open(path_param, "w")
        f.write(header)
        f.write("\n")

    f.write(line)
    f.write("\n")

    f.close()
    print("[+] ¡¡¡ SUCCESSFULLY line write!!! [+]")
    print("")
    print("")

def main(args):

    instantIni = datetime.now()

    root_path= './files/'
    mc_file = 'ugr16_multiclass.csv'
    mcfold_file = 'ugr16_multiclass_folds.csv'
    mcvars_file = 'ugr16_multiclass_folds_selecvars.csv'

    model=args.model
    rep=args.rep
    kfold=args.kfold

    df = pd.DataFrame(pd.read_csv(root_path + mc_file, index_col=0))
    df_folds = pd.DataFrame(pd.read_csv(root_path + mcfold_file, engine='python'))
    df_vars = pd.DataFrame(pd.read_csv(root_path + mcvars_file, engine='python'))
    print("[+] ¡¡Reading of datasets!! [+]")
    print("")
    print("")
    print("- " + mc_file + " SUCCESSFULLY!! [+]")
    print("")
    print("- " + mcfold_file + " SUCCESSFULLY!! [+]")
    print("")
    print("- " + mcvars_file + " SUCCESSFULLY!! [+]")
    print("")

    # Column categorical label numeric transformation

    df.loc[:, 'outcome.multiclass'] = df['outcome.multiclass'].map(
        {'background': 0, 'dos': 1, 'nerisbotnet': 2, 'scan': 3, 'sshscan': 4, 'udpscan': 5, 'spam': 6})

    # Feature selection

    d = (df_vars.groupby('repeticion').groups[rep]) & (df_vars.groupby('caja.de.test').groups[kfold])

    print("rep: ", rep)
    print("kfold: ", kfold)

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
    No_rows_fold = df_folds[df_folds[group] != kfold][group].index

    # Data TRAIN and LABEL
    X_train = X.drop(X.index[rows_fold])
    y_train = y.drop(y.index[rows_fold])
    y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6])

    # Data TEST and LABEL
    X_test = X.drop(X.index[No_rows_fold])
    y_test = y.drop(y.index[No_rows_fold])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])

    n_classes = y_train.shape[1]

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

    print("[+] Calculating hyperparameters for the classifier " + title + "[+]")
    print("")
    if model == 'rf':
        parameters = {'random_state': [0], 'n_estimators': [1, 10], 'criterion': ('gini', 'entropy'),
                      'max_depth': (1, 30)}
        model_grid = RandomForestClassifier()
    elif model == 'lr':
        # parameters = {'random_state':[0], 'C':[1,5,10,100], 'solver':('liblinear', 'lbfgs'), 'multi_class':('auto', 'ovr')}
        parameters = {'random_state': [0], 'C': [1, 2], 'solver': ('liblinear', 'lbfgs'), 'multi_class': ('auto', 'ovr')}
        model_grid = LogisticRegression()
    elif model == 'svc':
        parameters = {'gamma': [0.001, 0.01, 0.1, 1], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        model_grid = SVC()

    clf = GridSearchCV(model_grid, parameters, cv=5, verbose=100000)
    clf.fit(X_train_scaled, y_train)
    print("")
    print("[+] The best parameters for " + "Rep.: " + str(rep) + " and Kfold: " + str(kfold) + " are:  [+]")
    print(str(clf.best_params_))
    print("")
    bp = clf.best_params_

    # PARAMETERS SELECTED
    print("[+] PARAMETERS SELECTED MODEL " + title + " [+]")
    print("")
    if model == 'rf':
        cr = str(bp.get('criterion'))
        print("Criterion: ", cr)
        md = int(bp.get('max_depth'))
        print("Max_Depth: ", md)
        nit = int(bp.get('n_estimators'))
        print("N_Estimators: ", nit)
        tmodel = RandomForestClassifier(criterion=cr, max_depth=md, random_state=0, n_estimators=nit, verbose=100000)
    elif model == 'lr':
        cs = int(bp.get('C'))
        print("cs: ", cs)
        solv = str(bp.get('solver'))
        print("solv: ", solv)
        mc = str(bp.get('multi_class'))
        print("mc: ", mc)
        tmodel = LogisticRegression(random_state=0, C=cs, solver=solv, multi_class=mc, verbose=100000)
    elif model == 'svc':
        cs = int(bp.get('C'))
        print("cs: ", cs)
        ga = float(bp.get('gamma'))
        print("ga: ", ga)
        tmodel = SVC(random_state=0, kernel='rbf', gamma=ga, C=cs, verbose=100000)

    # Training models
    print("[+] Training models " + "[+]")

    print("[+] MODEL TRAINING " + model + "\n")
    tmodel.fit(X_train_scaled, y_train)
    print("")
    print("[+] MODEL PREDICTING " + model + "\n")
    predictions = tmodel.predict(X_test_scaled)
    print("")
    print("[+] CLASSIFICATION REPORT " + model + "\n")
    h = classification_report(y_test, predictions, output_dict=True,
                              target_names=['Background', 'Dos', 'Nerisbotnet', 'Scan', 'SSHscan', 'UDPscam', 'Spam'])
    print(classification_report(y_test, predictions,
                                target_names=['Background', 'Dos', 'Nerisbotnet', 'Scan', 'SSHscan', 'UDPscam',
                                              'Spam']))
    print("")

    # Store Classification Report into 'dict'
    result = defaultdict(list)
    for fr in h.values():
        if isinstance(fr, float):
            print("")
        else:
            for k, v in fr.items():
                result[k].append(v)

        # Store keys and values
    k = list(result.keys())
    v = list(result.values())

    # Compute ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute ROC area weighted
    supports_sum = 0
    auc_partial = 0
    for i in range(n_classes):
        supports_sum = supports_sum + (v[3][i])
        auc_partial = auc_partial + ((v[3][i]) * roc_auc[i])
    auc_w = auc_partial / supports_sum

    print("SUM SUPPORTS: ", supports_sum)
    print("AUC_PARTIAL: ", auc_partial)
    print("AUC_W: ", auc_w)
    print("")

    # Send data to .csv

    instantFinal = datetime.now()
    time = instantFinal - instantIni
    path_param = root_path + model + "_" + "results_" + ".csv"
    line = str(rep) + ',' + str(kfold) + ',' + str(len(f)) + ',' + str(v[0][0]) + ',' + str(v[1][0]) + ',' + str(
        v[2][0]) + ',' + str(roc_auc[0]) + ',' + str(v[0][1]) + ',' + str(v[1][1]) + ',' + str(v[2][1]) + ',' + str(
        roc_auc[1]) + ',' + str(v[0][2]) + ',' + str(v[1][2]) + ',' + str(v[2][2]) + ',' + str(roc_auc[2]) + ',' + str(
        v[0][3]) + ',' + str(v[1][3]) + ',' + str(v[2][3]) + ',' + str(roc_auc[3]) + ',' + str(v[0][4]) + ',' + str(
        v[1][4]) + ',' + str(v[2][4]) + ',' + str(roc_auc[4]) + ',' + str(v[0][5]) + ',' + str(v[1][5]) + ',' + str(
        v[2][5]) + ',' + str(roc_auc[5]) + ',' + str(v[0][6]) + ',' + str(v[1][6]) + ',' + str(v[2][6]) + ',' + str(
        roc_auc[6]) + ',' + str(v[0][8]) + ',' + str(v[1][8]) + ',' + str(v[2][8]) + ',' + str(auc_w) + ',' + str(time)
    header = "Rep." + "," + "Kfold" + "," + "Num. Vars." + "," + "Precision-Background" + "," + "Recall-Background" + "," + "F1_score_Background" + "," + "AUC_Background" + "," + "Precision-DoS" + "," + "Recall-DoS" + "," + "F1_score_DoS" + "," + "AUC_DoS" + "," + "Precision-Botnet" + "," + "Recall-Botnet" + "," + "F1_score_Botnet" + "," + "AUC_Botnet" + "," + "Precision-Scan" + "," + "Recall-Scan" + "," + "F1_score_Scan" + "," + "AUC_Scan" + "," + "Precision-SSHscan" + "," + "Recall-SSHscan" + "," + "F1_score_SSHscan" + "," + "AUC_SSHscan" + "," + "Precision-UDPscan" + "," + "Recall-UDPscan" + "," + "F1_score_UDPscan" + "," + "AUC_UDPscan" + "," + "Precision-Spam" + "," + "Recall-Spam" + "," + "F1_score_Spam" + "," + "AUC_Spam" + "," + "Precision-w" + "," + "Recall-w" + "," + "F1_score_w" + "," + "AUC_w" + "," + "Time"

    write_param(path_param, line, header)

    print("------------------")
    print(" [+] REP: ---" + str(rep) + "---" + " Kfold: " + "---" +  str(kfold) + "--- Model: ---" + title + "---" + " ¡¡TERMINATED!! [+]")
    print("------------------")
    print("Time elapsed: ", time)

if __name__ == "__main__":
    args = getArguments()

    main(args)
