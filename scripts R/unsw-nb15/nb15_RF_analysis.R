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
        #WORKING.DIR       = "/.../.../.../ff4ml/"
        BASE.RESULTS.PATH = "./results/"
        TYPE="ts"
        NOBS = 20000
        REP=1
        FOLD=1
}


#setwd(WORKING.DIR)



if (TYPE == "ts") {
     OUTFILE = paste0(BASE.RESULTS.PATH, "unsw-nb15/nb15_RF_", TYPE, "_results.csv")
     datos.nb15       = read.csv(file=paste0("./data/unsw-nb15/dat_ts/unsw-nb15_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
     folds.nb15       = read.csv(file=paste0("./data/unsw-nb15/dat_ts/unsw-nb15_multiclass_folds_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
     varselected.nb15 = read.csv(file=paste0("./data/unsw-nb15/dat_ts/unsw-nb15_multiclass_folds_selecvars_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)     
} else if (NOBS == 10000) {
     OUTFILE = paste0(BASE.RESULTS.PATH, "nb15_RF_", NOBS, "_results.csv")
     datos.nb15       = read.csv(file=paste0(WORKING.DIR, "data/unsw-nb15/dat_batches/output-NB15_all_extended_10000fobs_254bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
     folds.nb15       = read.csv(file=paste0(WORKING.DIR, "data/unsw-nb15/dat_batches/output-NB15_all_extended_10000fobs_254bsize_multiclass_folds_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
     varselected.nb15 = read.csv(file=paste0(WORKING.DIR, "data/unsw-nb15/dat_batches/output-NB15_all_extended_10000fobs_254bsize_multiclass_folds_selecvars_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
} else {
     OUTFILE = paste0(BASE.RESULTS.PATH, "nb15_RF_", NOBS, "_results.csv")
     datos.nb15       = read.csv(file=paste0(WORKING.DIR, "data/unsw-nb15/dat_batches/output-NB15_all_extended_20000fobs_127bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
     folds.nb15       = read.csv(file=paste0(WORKING.DIR, "data/unsw-nb15/dat_batches/output-NB15_all_extended_20000fobs_127bsize_multiclass_folds_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
     varselected.nb15 = read.csv(file=paste0(WORKING.DIR, "data/unsw-nb15/dat_batches/output-NB15_all_extended_20000fobs_127bsize_multiclass_folds_selecvars_20x5cv.csv"), header=T, sep=",", stringsAsFactors = F)
}
datos.nb15$outcome <- gsub("label_", "", datos.nb15$outcome)
if (TYPE=="ts") {
     idx.toremove      <- which(datos.nb15$outcome %in% c("dos"))
} else {
     idx.toremove      <- which(datos.nb15$outcome %in% c("analysis","backdoor"))    
}
if (length(idx.toremove > 0)) {
     datos.nb15         <- datos.nb15[-idx.toremove,]
     folds.nb15         <- folds.nb15[-idx.toremove,]
}

X        = datos.nb15[,1:134]
y        = as.factor(datos.nb15$outcome)
y       <- relevel(y, ref = "background") 

cat("% clase en todo el dataset NB15: ",  paste0("(", paste0(names(table(y)), collapse = ","), ")"), " = ", table(y)/length(y), "\n")

global <- NULL

for (REP in 1:20) {
     for (FOLD in 1:5) {
          idx.train       = folds.nb15[, REP] != FOLD
          idx.test        = folds.nb15[, REP] == FOLD
          idx.varselected = which((varselected.nb15[varselected.nb15$repeticion == REP & varselected.nb15$caja.de.test == FOLD, -c(1,2)]) != 0)

          X.train        = X[idx.train, idx.varselected]
          X.test         = X[idx.test, idx.varselected]
          y.train        = y[idx.train]
          y.test         = y[idx.test]
          
          # mean.x <- colMeans(X.train)
          # sd.x   <- apply(X.train, 2, sd)
          # idx.0sd <- which(sd.x == 0)
          # 
          # X.train <- t(apply(X.train, 1, function(x) (x - mean.x) / sd.x))
          # X.test  <- t(apply(X.test, 1, function(x) (x - mean.x) / sd.x))
          # if (length(idx.0sd > 0)) {
          #         X.train = X.train[,-idx.0sd]
          #         X.test  = X.test[,-idx.0sd]
          # }
          
          cat("[", REP,",", FOLD,"]: % clase en caja test: ",  paste0("(", paste0(names(table(y.test)), collapse = ","), ")"), " = ", table(y.test)/length(y.test), "\n")
          
          fitted.RF <- fit.RF(X = X.train, y = y.train)
          predclass.test <- get.predictions(fitted.RF, X.test, type = "response")
          probs.test     <- get.predictions(fitted.RF, X.test, type = "probs")
          
          global <- rbind(global, c(REP, FOLD, compute.auc.global(y.test, probs.test), compute.acc.global(y.test, predclass.test)))
     }
}

if (TYPE == "ts") {
     colnames(global) <- c("Repetition", "Fold", "auc.multiclass", "auc.background", "auc.exploit", "auc.fuzzer", "auc.generic", "auc.weighted", "acc.overall", "acc.background", "acc.exploit", "acc.fuzzer", "acc.generic")
} else if (NOBS == 10000) {
        colnames(global) <- c("Repetition", "Fold", "auc.multiclass", "auc.background", "auc.dos", "auc.exploit", "auc.fuzzer", "auc.generic", "auc.reconnaissance", "auc.weighted", "acc.overall", "acc.background", "acc.dos", "acc.exploit", "acc.fuzzer", "acc.generic", "acc.reconnaissance")
} else {
        colnames(global) <- c("Repetition", "Fold", "auc.multiclass", "auc.background", "auc.dos", "auc.exploit", "auc.fuzzer", "auc.generic", "auc.reconnaissance", "auc.shellcode", "auc.weighted", "acc.overall", "acc.background", "acc.dos", "acc.exploit", "acc.fuzzer", "acc.generic", "acc.reconnaissance", "acc.shellcode")
}

# global.res <- colMeans(global[,-c(1,2)])

write.table(matrix(global, nrow=1), file = OUTFILE, quote=F, col.names=F, row.names=F, append=TRUE, sep=",")
        
        