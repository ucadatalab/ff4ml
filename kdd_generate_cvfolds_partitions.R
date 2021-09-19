rm(list=ls())

library(caret)


NUMOBS   = "20000"
NUMREPS  = 5
NUMFOLDS = 2

dataset = read.csv(file=paste0("./data/nsl-kdd/dat_batches/kdd_all_extended_", NUMOBS, "fobs_multiclass.csv"), header=T, sep=",")

outcome    = dataset$outcome

set.seed(1234) #para replicar los resultados al relanzar
folds = sapply(1:NUMREPS, function(i) createFolds(y = factor(outcome), k=NUMFOLDS, list=F))
colnames(folds) = paste0("REP.", 1:NUMREPS)

write.csv(folds, file=paste0("./data/nsl-kdd/dat_batches/kdd_all_extended_", NUMOBS, "fobs_multiclass_folds_", NUMREPS, "x", NUMFOLDS, "cv.csv"), row.names=F)
