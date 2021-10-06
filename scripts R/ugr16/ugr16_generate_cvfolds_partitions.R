rm(list=ls())

library(caret)


TYPE     = "batches"     # ts ó batches
NUMOBS   = "20000"  # sólo se usa si TYPE = 'batches'
NUMREPS  = 5
NUMFOLDS = 2

if (TYPE == "ts") {
        dataset = read.csv(file=paste0("./data/ugr16/dat_", TYPE, "/ugr16_multiclass.csv"), header=T, sep=",")
} else {
        dataset = read.csv(file=paste0("./data/ugr16/dat_", TYPE, "/ugr16_all_extended_", NUMOBS, "fobs_multiclass.csv"), header=T, sep=",")
}

outcome    = dataset$outcome

set.seed(1234) #para replicar los resultados al relanzar
folds = sapply(1:NUMREPS, function(i) createFolds(y = factor(outcome), k=NUMFOLDS, list=F))
colnames(folds) = paste0("REP.", 1:NUMREPS)

if (TYPE == "ts") {
        write.csv(folds, file=paste0("./data/ugr16/dat_", TYPE, "/ugr16_multiclass_folds_", NUMREPS, "x", NUMFOLDS, "cv.csv"), row.names=F)
} else {
        write.csv(folds, file=paste0("./data/ugr16/dat_", TYPE, "/ugr16_all_extended_", NUMOBS, "fobs_multiclass_folds_", NUMREPS, "x", NUMFOLDS, "cv.csv"), row.names=F)
}