rm(list=ls())
library(caret)
library(ModelMetrics)
library(nnet)


setwd("/.../.../.../.../ff4ml/")



fit.MLR <- function(X, y) {
        datos    <- data.frame(X, outcome=y)
        fit_nnet <- multinom(outcome ~ ., data = datos, trace=F)
        fit_nnet
}


compute.auc.global <- function(y.test, probs.test) {
        result <- c()
        result <- c(result, pROC::multiclass.roc(y.test, probs.test)$auc[[1]])
        for (l in levels(y.test)) {
                y.test.bin <- rep(0, length(y.test))
                y.test.bin[y.test == l] <- 1
                result <- c(result, auc(y.test.bin, probs.test[,l]))
        }
        result <- c(result, sum(table(y.test)/length(y.test)*result[-1]))
        round(result, 3)
}


compute.acc.global <- function(y.test, predclass.test) {
        confmat <- confusionMatrix(data=y.test, reference=predclass.test)
        result  <- c(confmat$overall[1], as.numeric(confmat$byClass[,11]))
        result
}




NOBS    = 10000 #-> 10000 or 20000
OUTFILE = paste0(".../results/nsl-kdd/R execution/kdd_LR_", NOBS, "_results.csv")

if (NOBS == 10000) {
        datos.kdd       = read.csv(file=paste0(".../data/nsl-kdd/dat_batches/output-KDD_all_extended_10000fobs_14bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        folds.kdd       = read.csv(file=paste0(".../data/nsl-kdd/dat_batches/output-KDD_all_extended_10000fobs_14bsize_multiclass_folds_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
        varselected.kdd = read.csv(file=paste0(".../data/nsl-kdd/dat_batches/output-KDD_all_extended_10000fobs_14bsize_multiclass_folds_selecvars_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
} else {
        datos.kdd       = read.csv(file=paste0(".../data/nsl-kdd/dat_batches/output-KDD_all_extended_20000fobs_7bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        folds.kdd       = read.csv(file=paste0(".../data/nsl-kdd/dat_batches/output-KDD_all_extended_20000fobs_7bsize_multiclass_folds_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
        varselected.kdd = read.csv(file=paste0(".../data/nsl-kdd/dat_batches/output-KDD_all_extended_20000fobs_7bsize_multiclass_folds_selecvars_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
}
datos.kdd$outcome <- gsub("label_category_", "", datos.kdd$outcome)
if (NOBS==10000) {
        idx.toremove      <- which(datos.kdd$outcome %in% c("u2r"))
        datos.kdd         <- datos.kdd[-idx.toremove,]
        folds.kdd         <- folds.kdd[-idx.toremove,]
}

X        = datos.kdd[,1:134]
y        = as.factor(datos.kdd$outcome)
y       <- relevel(y, ref = "background") 

cat("% clase en todo el dataset KDD: ",  paste0("(", paste0(names(table(y)), collapse = ","), ")"), " = ", table(y)/length(y), "\n")

global <- NULL
for (REP in 1:20) {
        for (FOLD in 1:5) {
                idx.train       = folds.kdd[, REP] != FOLD
                idx.test        = folds.kdd[, REP] == FOLD
                idx.varselected = which((varselected.kdd[varselected.kdd$repeticion == REP & varselected.kdd$caja.de.test == FOLD, -c(1,2)]) != 0)
                
                X.train        = X[idx.train, idx.varselected]
                X.test         = X[idx.test, idx.varselected]
                y.train        = y[idx.train]
                y.test         = y[idx.test]
                
                mean.x <- colMeans(X.train)
                sd.x   <- apply(X.train, 2, sd)
                idx.0sd <- which(sd.x == 0)
                
                X.train <- t(apply(X.train, 1, function(x) (x - mean.x) / sd.x))
                X.test  <- t(apply(X.test, 1, function(x) (x - mean.x) / sd.x))
                if (length(idx.0sd > 0)) {
                        X.train = X.train[,-idx.0sd]
                        X.test  = X.test[,-idx.0sd]
                }
                
                cat("[", REP,",", FOLD,"]: % clase en caja test: ",  paste0("(", paste0(names(table(y.test)), collapse = ","), ")"), " = ", table(y.test)/length(y.test), "\n")
                fitted.MLR <- fit.MLR(X = X.train, y = y.train)
                predclass.test <- predict(fitted.MLR, newdata = X.test)
                probs.test     <- predict(fitted.MLR, newdata = X.test, "probs")
                
                global <- rbind(global, c(REP, FOLD, compute.auc.global(y.test, probs.test), compute.acc.global(y.test, predclass.test)))
        }
}
if (NOBS == 10000) {
        colnames(global) <- c("Repetition", "Fold", "auc.multiclass", "auc.background", "auc.dos", "auc.probe", "auc.r2l", "auc.weighted", "acc.overall", "acc.background", "acc.dos", "acc.probe", "acc.r2l")
} else {
        colnames(global) <- c("Repetition", "Fold", "auc.multiclass", "auc.background", "auc.dos", "auc.probe", "auc.r2l", "auc.u2r", "auc.weighted", "acc.overall", "acc.background", "acc.dos", "acc.probe", "acc.r2l", "acc.u2r")
}

# global.res <- colMeans(global[,-c(1,2)])

write.csv(global, file = OUTFILE, row.names = F)
        
        