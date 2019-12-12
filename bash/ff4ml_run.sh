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
  echo "<model> can be: svc, rf, lr"
  echo " "
  exit 1
fi

# Notifications
n_to=ignacio.diaz@uca.es

# model
model=$1

# repetitions
REP_START=1
REP_END=20

# K-folds
OUTFOLD_START=1
OUTFOLD_END=5

# execution timestamp
exec_ts=`date +"%Y%m%d_%H%M%S"`

# make the execution folder
mkdir -p ../results/$exec_ts

# make the same execution folder for the logs
mkdir -p ../logs/$exec_ts

#To run each task in a whole CPU
#FLAGS="--job-name="MDPI_FE" --exclusive --cpus-per-task=1 --time=7-00:00:00 --mem=16GB --error=../logs/job.%J.err --output=../logs/job.%J.out"

#To run each task in just one core of a CPU
FLAGS="--job-name=MDPI$model --tasks=1 --time=7-00:00:00 --mem=16GB --mail-user=$n_to --mail-type=END,FAIL,TIME_LIMIT_80 --error=../logs/$exec_ts/job.%J.err --output=../logs/$exec_ts/job.%J.out"

TOTAL=0

echo "[+] Running experiment $exec_ts with rep ($REP_START,$REP_END) and k-folds ($OUTFOLD_START,$OUTFOLD_END)"

for ((r=$REP_START; r<=$REP_END; r++))
do
    for ((k=$OUTFOLD_START; k<=$OUTFOLD_END; k++))
    do
	echo "[-] Running task for model: $model, rep: $r, k-fold: $k"
        sbatch $FLAGS --wrap="python ../main.py $model $r $k $exec_ts"
        # python ../main.py $model $r $k $exec_ts
        TOTAL=`expr $TOTAL + 1`
        sleep 2s
    done
done

echo "[+] $TOTAL repetitions have been launched for the experiment $exec_ts"

