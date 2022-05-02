rm(list=ls())

library(caret)

# #código para generar la variable outcome: SOLO ejecutar una sola vez al principio, pero dejo el codigo aqui comentado
# preprocess.outcome <- function(x) {
#      if (all(x[-1] == 0)) {
#           clase = names(x)[1]  #background!!
#      } else {
#           clase = names(x[-1])[which.max(x[-1])]
#           # if (clase %in% c("label_scan44","label_scan11")) {
#           #      clase = "label_scan"
#           # }
#      }
#      
#      return(clase)
# }
# 
# setwd("./attackclass/ff4ml/")
# aaa = read.csv(file="./data/unsw-nb15//dat_ts/unsw-nb15.csv", header = T)
# idx.labels = grep("^label", colnames(aaa))
# idx.labels = idx.labels[-11] #ignoramos la columna label_attack
# outcome    = apply(aaa[,idx.labels], 1, preprocess.outcome)


TYPE     = "ts"     # ts ó batches
NUMOBS   = "20000"
NUMREPS  = 20
NUMFOLDS = 5

if (TYPE == "ts") {
     dataset = read.csv(file=paste0("./data/unsw-nb15/dat_", TYPE, "/unsw-nb15_multiclass.csv"), header=T, sep=",")
} else {
     dataset = read.csv(file=paste0("./data/unsw-nb15/dat_", TYPE, "/nb15_all_extended_", NUMOBS, "fobs_multiclass.csv"), header=T, sep=",")
}

outcome    = dataset$outcome

set.seed(1234) #para replicar los resultados al relanzar
folds = sapply(1:NUMREPS, function(i) createFolds(y = factor(outcome), k=NUMFOLDS, list=F))
colnames(folds) = paste0("REP.", 1:NUMREPS)

if (TYPE == "ts") {
     write.csv(folds, file=paste0("./data/unsw-nb15/dat_", TYPE, "/unsw-nb15_multiclass_folds_", NUMREPS, "x", NUMFOLDS, "cv.csv"), row.names=F)
} else {
     write.csv(folds, file=paste0("./data/unsw-nb15/dat_", TYPE, "/nb15_all_extended_", NUMOBS, "fobs_multiclass_folds_", NUMREPS, "x", NUMFOLDS, "cv.csv"), row.names=F)
}
