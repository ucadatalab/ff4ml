#!/bin/bash

# 
# Authors: 
# 	- Roberto Magán Carrión - roberto.magan@uca.es, rmagan@ugr.es
# UCADatalab - http://datalab.uca.es , NESG(Network Engineering & Security Group) - https://nesg.ugr.es
# Date: 2019/12/11
#

if [ "$#" -ne 1 ]; then
  echo " "
  echo "Use: bash ff4ml_run.sh <model>"
  echo "Example of use:"
  echo "<model> can be: svm, rf, lr"
  echo " "
  exit 1
fi

# model
model=$1

# repetitions
rep=20

# k-folds
kfold=5

for r in $(seq 1 $rep);
do
	for k in $(seq 1 $kfold);
	do
		echo "[+] Running experiment -- Rep: $r, K-fold: $k"
		python ../main.py $model $r $k
	done
done

