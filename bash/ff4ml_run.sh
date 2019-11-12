#!/bin/bash



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

