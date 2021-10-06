rm(list=ls())
library(caret)
library(ModelMetrics)
library(nnet)


setwd("/.../.../.../.../ff4ml/")



fit.MLR <- function(X, y) {
        datos    <- data.frame(X, outcome=y)
        fit_nnet <- nnet::multinom(formula = outcome ~ ., data = datos, trace=F)
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
        confmat <- caret::confusionMatrix(data=y.test, reference=predclass.test)
        result  <- c(confmat$overall[1], as.numeric(confmat$byClass[,11]))
        round(result, 3)
}




NOBS = 10000 #-> 10000 or 20000
OUTFILE = paste0(".../results/crossed_LR_", NOBS, "_results.csv")

if (NOBS == 10000) {
        datos.ugr16       = read.csv(file=paste0(".../data/ugr16/dat_batches/output-UGR16_all_extended_10000fobs_395082bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.ugr16       = datos.ugr16[,c(1:134, 144)]
        datos.ugr16$outcome = gsub("label_", "", datos.ugr16$outcome)
        idx.common.outcome  = which(datos.ugr16$outcome %in% c("background", "dos", "scan"))
        datos.ugr16       = datos.ugr16[idx.common.outcome,]
        varselected.ugr16 = read.csv(file=paste0(".../data/mix/rowwise/dat_batches/ugr16_all_extended_10000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        ugr16.norm        = 395082
        
        datos.nb15       = read.csv(file=paste0(".../data/unsw-nb15/dat_batches/output-NB15_all_extended_10000fobs_254bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.nb15       = datos.nb15[,c(1:134, 146)]
        datos.nb15$outcome = gsub("label_", "", datos.nb15$outcome)
        datos.nb15$outcome = gsub("reconnaissance", "scan", datos.nb15$outcome)
        idx.common.outcome  = which(datos.nb15$outcome %in% c("background", "dos", "scan"))
        datos.nb15       = datos.nb15[idx.common.outcome,]
        varselected.nb15 = read.csv(file=paste0(".../data/mix/rowwise/dat_batches/nb15_all_extended_10000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        nb15.norm        = 254
        
        datos.kdd       = read.csv(file=paste0(".../data/nsl-kdd/dat_batches/output-KDD_all_extended_10000fobs_14bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.kdd       = datos.kdd[,c(1:134, 163)]
        datos.kdd$outcome = gsub("label_category_", "", datos.kdd$outcome)
        datos.kdd$outcome = gsub("probe", "scan", datos.kdd$outcome)
        idx.common.outcome  = which(datos.kdd$outcome %in% c("background", "dos", "scan"))
        datos.kdd       = datos.kdd[idx.common.outcome,]
        varselected.kdd = read.csv(file=paste0(".../data/mix/rowwise/dat_batches/kdd_all_extended_10000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        kdd.norm        = 14
} else {
        datos.ugr16       = read.csv(file=paste0(".../data/ugr16/dat_batches/output-UGR16_all_extended_20000fobs_197541bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.ugr16       = datos.ugr16[,c(1:134, 144)]
        datos.ugr16$outcome = gsub("label_", "", datos.ugr16$outcome)
        idx.common.outcome  = which(datos.ugr16$outcome %in% c("background", "dos", "scan"))
        datos.ugr16       = datos.ugr16[idx.common.outcome,]
        varselected.ugr16 = read.csv(file=paste0(".../data/mix/rowwise/dat_batches/ugr16_all_extended_20000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        ugr16.norm        = 197541
        
        datos.nb15       = read.csv(file=paste0(".../data/unsw-nb15/dat_batches/output-NB15_all_extended_20000fobs_127bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.nb15       = datos.nb15[,c(1:134, 146)]
        datos.nb15$outcome = gsub("label_", "", datos.nb15$outcome)
        datos.nb15$outcome = gsub("reconnaissance", "scan", datos.nb15$outcome)
        idx.common.outcome  = which(datos.nb15$outcome %in% c("background", "dos", "scan"))
        datos.nb15       = datos.nb15[idx.common.outcome,]
        varselected.nb15 = read.csv(file=paste0(".../data/mix/rowwise/dat_batches/nb15_all_extended_20000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        nb15.norm        = 127
        
        datos.kdd       = read.csv(file=paste0(".../data/nsl-kdd/dat_batches/output-KDD_all_extended_20000fobs_7bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.kdd       = datos.kdd[,c(1:134, 163)]
        datos.kdd$outcome = gsub("label_category_", "", datos.kdd$outcome)
        datos.kdd$outcome = gsub("probe", "scan", datos.kdd$outcome)
        idx.common.outcome  = which(datos.kdd$outcome %in% c("background", "dos", "scan"))
        datos.kdd       = datos.kdd[idx.common.outcome,]
        varselected.kdd = read.csv(file=paste0(".../data/mix/rowwise/dat_batches/kdd_all_extended_20000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        kdd.norm        = 7
}


trials = cbind(train=c("nb15", "kdd", "ugr16", "kdd", "ugr16", "nb15"),
               test=c("ugr16", "ugr16", "nb15", "nb15", "kdd", "kdd"))

global <- NULL
for (i in 1:nrow(trials)) {
        if (trials[i,1] == "ugr16") {
                data.train   = datos.ugr16
                varselection = varselected.ugr16
                train.norm.factor  = ugr16.norm
        } else if (trials[i,1] == "nb15") {
                data.train   = datos.nb15
                varselection = varselected.nb15
                train.norm.factor  = nb15.norm
        } else {
                data.train   = datos.kdd
                varselection = varselected.kdd
                train.norm.factor  = kdd.norm
        }
        
        if (trials[i,2] == "ugr16") {
                data.test   = datos.ugr16
                test.norm.factor  = ugr16.norm
        } else if (trials[i,2] == "nb15") {
                data.test   = datos.nb15
                test.norm.factor  = nb15.norm
        } else {
                data.test   = datos.kdd
                test.norm.factor  = kdd.norm
        }
        
        
        idx.varselected = which(varselection != 0)
        
        X.train        = data.train[, 1:134]
        X.train        = X.train[, idx.varselected]
        X.test         = data.test[, 1:134]
        X.test         = X.test[, idx.varselected]
        y.train        = as.factor(data.train$outcome)
        y.train        = relevel(y.train, ref = "background")
        y.test         = as.factor(data.test$outcome)
        
        X.train = X.train/train.norm.factor
        X.test  = X.test/test.norm.factor
        
        mean.x  <- colMeans(X.train)
        sd.x    <- apply(X.train, 2, sd)
        idx.0sd <- which(sd.x == 0)
        
        X.train <- t(apply(X.train, 1, function(x) (x - mean.x) / sd.x))
        X.test  <- t(apply(X.test, 1, function(x) (x - mean.x) / sd.x))
        if (length(idx.0sd > 0)) {
                X.train = X.train[,-idx.0sd]
                X.test  = X.test[,-idx.0sd]
        }
        
        cat("[", trials[i,1],"->", trials[i,2],"]:\n% clase en caja train: ",  paste0("(", paste0(names(table(y.train)), collapse = ","), ")"), " = ", table(y.train)/length(y.train), "\n% clase en caja test: ",  paste0("(", paste0(names(table(y.test)), collapse = ","), ")"), " = ", table(y.test)/length(y.test), "\n")
        
        fitted.MLR <- fit.MLR(X = X.train, y = y.train)
        predclass.test <- predict(fitted.MLR, newdata = X.test)
        probs.test     <- predict(fitted.MLR, newdata = X.test, "probs")
        
        global <- rbind(global, c(trials[i,1], trials[i,2], compute.auc.global(y.test, probs.test), compute.acc.global(y.test, predclass.test)))
}

colnames(global) <- c("Train.dataset", "Test.dataset", "auc.multiclass", "auc.background", "auc.dos", "auc.scan", "auc.weighted", "acc.overall", "acc.background", "acc.dos", "acc.scan")


# global.res <- colMeans(global[,-c(1,2)])

write.csv(global, file = OUTFILE, row.names = F)
        
        