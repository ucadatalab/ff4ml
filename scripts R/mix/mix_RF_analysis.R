rm(list=ls())
library(caret)
library(mlr)
library(mlrMBO)
library(ModelMetrics)
library(nnet)


#
train.bayopt.rf <- function (X, y, resampling = mlr::cv3,
                             iters = 10, crit = makeMBOInfillCritEI(),
                             ntree.lower = 50, ntree.upper = 2000,
                             mtry.lower = 2, mtry.upper = max(ceiling(sqrt(ncol(X))), min(10, ncol(X))),
                             nodesize.lower = 5, nodesize.upper = 150,
                             show.info = F) {
        if (is.factor(y) && length(levels(y))==2) {
                df = cbind(data.frame(X), Outcome = y)
                learner.name = "classif.randomForestSRC"
                data.task = makeClassifTask(data = df, target = "outcome", positive = "1")
        } else if (is.factor(y)){
                df = cbind(data.frame(X), nnet::class.ind(y))
                targets = levels(y)
                df[,targets] = apply(df[,targets], 2, as.logical)
                learner.name = "multilabel.randomForestSRC"
                data.task = makeMultilabelTask(data = df, target = targets)
        } else {
                df = cbind(data.frame(X), Outcome = y)
                learner.name = "regr.randomForestSRC"
                data.task = makeRegrTask(data = df, target = "outcome")
        }
        
        par.set = makeParamSet(
                #makeIntegerParam("ntree", ntree.lower, ntree.upper),
                makeIntegerParam("mtry", mtry.lower, mtry.upper),
                makeIntegerParam("nodesize", lower = nodesize.lower, upper = nodesize.upper)
        )
        
        
        ctrl = makeMBOControl()
        ctrl = setMBOControlTermination(ctrl, iters = iters)
        ctrl = setMBOControlInfill(ctrl, crit = crit)
        tune.ctrl = makeTuneControlMBO(mbo.control = ctrl)
        
        if (is.factor(y)) {
                lrn = makeLearner(learner.name, predict.type = "prob", ntree = 1000)
                res = tuneParams(learner = lrn,
                                 task = data.task, 
                                 resampling = resampling, 
                                 par.set = par.set, 
                                 control = tune.ctrl, 
                                 #measures = mlr::acc,
                                 show.info = show.info)
        } else {
                lrn = makeLearner(learner.name, ntree = 1000)
                res = tuneParams(learner = lrn, 
                                 task = data.task, 
                                 resampling = resampling, 
                                 par.set = par.set, 
                                 control = tune.ctrl, 
                                 measures = mlr::rmse, show.info = show.info)
        }
        
        if (show.info) print(res)
        
        lrn = setHyperPars(lrn, par.vals = res$x)
        
        mlr::train(learner = lrn, task = data.task)
}


fit.RF <- function(X, y) {
        fit_rf = train.bayopt.rf(X=X, y=y, show.info = T)
        fit_rf
}


get.predictions <- function(fitted.model, X, type) {
        predictions = predict(fitted.model,  newdata = X)
        if (type == "probs") {
                probs = getPredictionProbabilities(predictions)
                predictions = probs
        } else {
                probs = getPredictionProbabilities(predictions)
                response = getPredictionResponse(predictions)
                
                idx.sinrespuesta = which(apply(response, 1, sum)!=1)
                if (length(idx.sinrespuesta) > 0) {
                        resps.a.incluir = colnames(probs)[apply(probs[idx.sinrespuesta,], 1, which.max)]
                        for (i in 1:length(idx.sinrespuesta)) {
                                next.id = idx.sinrespuesta[i]
                                response[next.id, resps.a.incluir[i]] = T
                        }
                }
                predictions = as.factor(colnames(response)[apply(response, 1, which.max)])
        }
        return(predictions)
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
        if (length(levels(predclass.test)) != length(levels(y.test))) {
                to.add <- setdiff(levels(y.test), levels(predclass.test))
                levels(predclass.test) = sort(c(levels(predclass.test), to.add))
        }
        confmat <- caret::confusionMatrix(data=y.test, reference=predclass.test)
        result  <- c(confmat$overall[1], as.numeric(confmat$byClass[,11]))
        result
}


compute.auc.ugr16 <- function(y.test, probs.test, origdata.test) {
        result <- c()
        idx.ugr <- which(origdata.test == "ugr16")
        
        probs.test.ugr <- probs.test[idx.ugr,]
        y.test.ugr     <- y.test[idx.ugr]
        
        result <- c(result, pROC::multiclass.roc(y.test.ugr, probs.test.ugr)$auc[[1]])
        for (l in levels(y.test)) {
                if (sum(y.test.ugr == l) == 0) {
                        result <- c(result, NA)
                } else { 
                        y.test.ugr.bin <- rep(0, length(y.test.ugr))
                        y.test.ugr.bin[y.test.ugr == l] <- 1
                        result <- c(result, auc(y.test.ugr.bin, probs.test.ugr[,l]))
                }
        }
        
        result <- c(result, sum(table(y.test.ugr)/length(y.test.ugr)*result[-1]))
        round(result, 3)
}


compute.acc.ugr16 <- function(y.test, predclass.test) {
        idx.ugr <- which(origdata.test == "ugr16")
        
        predclass.test.ugr <- predclass.test[idx.ugr]
        y.test.ugr         <- y.test[idx.ugr]
        
        confmat <- caret::confusionMatrix(data=y.test.ugr, reference=predclass.test.ugr)
        result  <- c(confmat$overall[1], as.numeric(confmat$byClass[,11]))
        result
}


compute.auc.nb15 <- function(y.test, probs.test, origdata.test) {
        result <- c()
        idx.nb15 <- which(origdata.test == "unsw-nb15")
        
        probs.test.nb15 <- probs.test[idx.nb15,]
        y.test.nb15     <- y.test[idx.nb15]
        
        result <- c(result, pROC::multiclass.roc(y.test.nb15, probs.test.nb15)$auc[[1]])
        for (l in levels(y.test)) {
                if (sum(y.test.nb15 == l) == 0) {
                        result <- c(result, NA)
                } else {
                        y.test.nb15.bin <- rep(0, length(y.test.nb15))
                        y.test.nb15.bin[y.test.nb15 == l] <- 1
                        result <- c(result, auc(y.test.nb15.bin, probs.test.nb15[,l]))
                }
        }
        
        result <- c(result, sum(table(y.test.nb15)/length(y.test.nb15)*result[-1]))
        round(result, 3)
}


compute.acc.nb15 <- function(y.test, predclass.test) {
        idx.nb15 <- which(origdata.test == "unsw-nb15")
        
        predclass.test.nb15 <- predclass.test[idx.nb15]
        y.test.nb15         <- y.test[idx.nb15]
        
        confmat <- caret::confusionMatrix(data=y.test.nb15, reference=predclass.test.nb15)
        result  <- c(confmat$overall[1], as.numeric(confmat$byClass[,11]))
        result
}


compute.auc.kdd <- function(y.test, probs.test, origdata.test) {
        result <- c()
        idx.kdd <- which(origdata.test == "nsl-kdd")
        
        probs.test.kdd <- probs.test[idx.kdd,]
        y.test.kdd     <- y.test[idx.kdd]
        
        result <- c(result, pROC::multiclass.roc(y.test.kdd, probs.test.kdd)$auc[[1]])
        for (l in levels(y.test)) {
                if (sum(y.test.kdd == l) == 0) {
                        result <- c(result, NA)
                } else {
                        y.test.kdd.bin <- rep(0, length(y.test.kdd))
                        y.test.kdd.bin[y.test.kdd == l] <- 1
                        result <- c(result, auc(y.test.kdd.bin, probs.test.kdd[,l]))
                }
        }
        
        result <- c(result, sum(table(y.test.kdd)/length(y.test.kdd)*result[-1]))
        round(result, 3)
}


compute.acc.kdd <- function(y.test, predclass.test) {
        idx.kdd <- which(origdata.test == "nsl-kdd")
        
        predclass.test.kdd <- predclass.test[idx.kdd]
        y.test.kdd         <- y.test[idx.kdd]
        
        confmat <- caret::confusionMatrix(data=y.test.kdd, reference=predclass.test.kdd)
        result  <- c(confmat$overall[1], as.numeric(confmat$byClass[,11]))
        result
}









IS.CLUSTER  = FALSE

if (IS.CLUSTER) {
        WORKING.DIR       = "/.../.../.../ff4ml/"
        BASE.RESULTS.PATH = "/.../.../.../ff4ml/results/"
        Sys.unsetenv("http_proxy")
        Sys.setenv("MC_CORES"=16L) 
        #get partition ID from SARRAY
        args = commandArgs(TRUE)
        NOBS = as.numeric(args[1])
        REP  = as.numeric(args[2])
        FOLD = as.numeric(args[3])
} else {
        WORKING.DIR       = "/.../.../.../ff4ml/"
        BASE.RESULTS.PATH = "/.../.../.../ff4ml/results/"
        NOBS = 20000
        REP=1
        FOLD=1
}


setwd(WORKING.DIR)

OUTFILE.MIXMIX   = paste0(BASE.RESULTS.PATH, "mix-mix_RF_", NOBS, "_results.csv")
OUTFILE.MIXUGR16 = paste0(BASE.RESULTS.PATH, "mix-ugr16_RF_", NOBS, "_results.csv")
OUTFILE.MIXNB15 = paste0(BASE.RESULTS.PATH, "mix-nb15_RF_", NOBS, "_results.csv")
OUTFILE.MIXKDD = paste0(BASE.RESULTS.PATH, "mix-kdd_RF_", NOBS, "_results.csv")

datos.mix       = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/output-rowise_", NOBS, "fobs_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
folds.mix       = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/output-rowise_", NOBS, "fobs_multiclass_folds_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
varselected.mix = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/output-rowise_", NOBS, "fobs_multiclass_folds_selecvars_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)

X        = datos.mix[,1:134]
y        = as.factor(datos.mix$outcome)
y       <- relevel(y, ref = "background") 
origdata = datos.mix$dataset

cat("% clase en todo el dataset mix: ",  paste0("(", paste0(names(table(y)), collapse = ","), ")"), " = ", table(y)/length(y), "\n")
cat("% muestras presentes de cada dataset:",   paste0("(", paste0(names(table(origdata)), collapse = ","), ")"), " = ", table(origdata)/length(origdata), "\n")

global <- NULL
ugr16  <- NULL
nb15  <- NULL
kdd  <- NULL

idx.train       = folds.mix[, REP] != FOLD
idx.test        = folds.mix[, REP] == FOLD
idx.varselected = which((varselected.mix[varselected.mix$repeticion == REP & varselected.mix$caja.de.test == FOLD, -c(1,2)]) != 0)

X.train        = X[idx.train, idx.varselected]
X.test         = X[idx.test, idx.varselected]
y.train        = y[idx.train]
y.test         = y[idx.test]
origdata.train = origdata[idx.train]
origdata.test  = origdata[idx.test]

if (NOBS == 10000) {
        X.train[origdata.train=="ugr16",]     = X.train[origdata.train=="ugr16",]/395082
        X.train[origdata.train=="unsw-nb15",] = X.train[origdata.train=="unsw-nb15",]/254
        X.train[origdata.train=="nsl-kdd",]   = X.train[origdata.train=="nsl-kdd",]/14
        X.test[origdata.test=="ugr16",]      = X.test[origdata.test=="ugr16",]/395082
        X.test[origdata.test=="unsw-nb15",]  = X.test[origdata.test=="unsw-nb15",]/254
        X.test[origdata.test=="nsl-kdd",]    = X.test[origdata.test=="nsl-kdd",]/14
} else {
        X.train[origdata.train=="ugr16",]     = X.train[origdata.train=="ugr16",]/197541
        X.train[origdata.train=="unsw-nb15",] = X.train[origdata.train=="unsw-nb15",]/127
        X.train[origdata.train=="nsl-kdd",]   = X.train[origdata.train=="nsl-kdd",]/7
        X.test[origdata.test=="ugr16",]      = X.test[origdata.test=="ugr16",]/197541
        X.test[origdata.test=="unsw-nb15",]  = X.test[origdata.test=="unsw-nb15",]/127
        X.test[origdata.test=="nsl-kdd",]    = X.test[origdata.test=="nsl-kdd",]/7
}

# mean.x <- colMeans(X.train)
# sd.x   <- apply(X.train, 2, sd)
# 
# X.train <- t(apply(X.train, 1, function(x) (x - mean.x) / sd.x))
# X.test  <- t(apply(X.test, 1, function(x) (x - mean.x) / sd.x))

cat("[", REP,",", FOLD,"]: % clase en caja test: ",  paste0("(", paste0(names(table(y.test)), collapse = ","), ")"), " = ", table(y.test)/length(y.test), "\n")

fitted.RF <- fit.RF(X = X.train, y = y.train)
predclass.test <- get.predictions(fitted.RF, X.test, type = "response")
probs.test     <- get.predictions(fitted.RF, X.test, type = "probs")

global <- rbind(global, c(REP, FOLD, compute.auc.global(y.test, probs.test), compute.acc.global(y.test, predclass.test)))
ugr16  <- rbind(ugr16, c(REP, FOLD, compute.auc.ugr16(y.test, probs.test, origdata.test), compute.acc.ugr16(y.test, predclass.test)))
nb15   <- rbind(nb15, c(REP, FOLD, compute.auc.nb15(y.test, probs.test, origdata.test), compute.acc.nb15(y.test, predclass.test)))
kdd    <- rbind(kdd, c(REP, FOLD, compute.auc.kdd(y.test, probs.test, origdata.test), compute.acc.kdd(y.test, predclass.test)))

colnames(global) <- c("Repetition", "Fold", "auc.multiclass", "auc.background", "auc.dos", "auc.scan", "auc.weighted", "acc.overall", "acc.background", "acc.dos", "acc.scan")
colnames(ugr16)  <- colnames(global)
colnames(nb15)   <- colnames(global)
colnames(kdd)    <- colnames(global)

# global.res <- colMeans(global[,-c(1,2)])
# ugr16.res  <- colMeans(ugr16[,-c(1,2)], na.rm = T)
# nb15.res   <- colMeans(nb15[,-c(1,2)], na.rm = T)
# kdd.res    <- colMeans(kdd[,-c(1,2)], na.rm = T)

write.table(matrix(global, nrow=1), file = OUTFILE.MIXMIX, quote=F, col.names=F, row.names=F, append=TRUE, sep=",")
write.table(matrix(ugr16, nrow=1), file = OUTFILE.MIXUGR16, quote=F, col.names=F, row.names=F, append=TRUE, sep=",")
write.table(matrix(nb15, nrow=1), file = OUTFILE.MIXNB15, quote=F, col.names=F, row.names=F, append=TRUE, sep=",")
write.table(matrix(kdd, nrow=1), file = OUTFILE.MIXKDD, quote=F, col.names=F, row.names=F, append=TRUE, sep=",")
        
        