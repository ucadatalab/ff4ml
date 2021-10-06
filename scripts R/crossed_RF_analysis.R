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
        args  = commandArgs(TRUE)
        NOBS  = as.numeric(args[1])
        i = as.numeric(args[2])
} else {
        WORKING.DIR       = "/.../.../.../ff4ml/"
        BASE.RESULTS.PATH = "/.../.../.../ff4ml/results/"
        NOBS = 10000
        i=1
}


setwd(WORKING.DIR)

OUTFILE = paste0(BASE.RESULTS.PATH, "crossed_RF_", NOBS, "_results.csv")

if (NOBS == 10000) {
        datos.ugr16       = read.csv(file=paste0(WORKING.DIR, "data/ugr16/dat_batches/output-UGR16_all_extended_10000fobs_395082bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.ugr16       = datos.ugr16[,c(1:134, 144)]
        datos.ugr16$outcome = gsub("label_", "", datos.ugr16$outcome)
        idx.common.outcome  = which(datos.ugr16$outcome %in% c("background", "dos", "scan"))
        datos.ugr16       = datos.ugr16[idx.common.outcome,]
        varselected.ugr16 = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/ugr16_all_extended_10000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        ugr16.norm        = 395082
        
        datos.nb15       = read.csv(file=paste0(WORKING.DIR, "data/unsw-nb15/dat_batches/output-NB15_all_extended_10000fobs_254bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.nb15       = datos.nb15[,c(1:134, 146)]
        datos.nb15$outcome = gsub("label_", "", datos.nb15$outcome)
        datos.nb15$outcome = gsub("reconnaissance", "scan", datos.nb15$outcome)
        idx.common.outcome  = which(datos.nb15$outcome %in% c("background", "dos", "scan"))
        datos.nb15       = datos.nb15[idx.common.outcome,]
        varselected.nb15 = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/nb15_all_extended_10000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        nb15.norm        = 254
        
        datos.kdd       = read.csv(file=paste0(WORKING.DIR, "data/nsl-kdd/dat_batches/output-KDD_all_extended_10000fobs_14bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.kdd       = datos.kdd[,c(1:134, 163)]
        datos.kdd$outcome = gsub("label_category_", "", datos.kdd$outcome)
        datos.kdd$outcome = gsub("probe", "scan", datos.kdd$outcome)
        idx.common.outcome  = which(datos.kdd$outcome %in% c("background", "dos", "scan"))
        datos.kdd       = datos.kdd[idx.common.outcome,]
        varselected.kdd = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/kdd_all_extended_10000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        kdd.norm        = 14
} else {
        datos.ugr16       = read.csv(file=paste0(WORKING.DIR, "data/ugr16/dat_batches/output-UGR16_all_extended_20000fobs_197541bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.ugr16       = datos.ugr16[,c(1:134, 144)]
        datos.ugr16$outcome = gsub("label_", "", datos.ugr16$outcome)
        idx.common.outcome  = which(datos.ugr16$outcome %in% c("background", "dos", "scan"))
        datos.ugr16       = datos.ugr16[idx.common.outcome,]
        varselected.ugr16 = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/ugr16_all_extended_20000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        ugr16.norm        = 197541
        
        datos.nb15       = read.csv(file=paste0(WORKING.DIR, "data/unsw-nb15/dat_batches/output-NB15_all_extended_20000fobs_127bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.nb15       = datos.nb15[,c(1:134, 146)]
        datos.nb15$outcome = gsub("label_", "", datos.nb15$outcome)
        datos.nb15$outcome = gsub("reconnaissance", "scan", datos.nb15$outcome)
        idx.common.outcome  = which(datos.nb15$outcome %in% c("background", "dos", "scan"))
        datos.nb15       = datos.nb15[idx.common.outcome,]
        varselected.nb15 = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/nb15_all_extended_20000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        nb15.norm        = 127
        
        datos.kdd       = read.csv(file=paste0(WORKING.DIR, "data/nsl-kdd/dat_batches/output-KDD_all_extended_20000fobs_7bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.kdd       = datos.kdd[,c(1:134, 163)]
        datos.kdd$outcome = gsub("label_category_", "", datos.kdd$outcome)
        datos.kdd$outcome = gsub("probe", "scan", datos.kdd$outcome)
        idx.common.outcome  = which(datos.kdd$outcome %in% c("background", "dos", "scan"))
        datos.kdd       = datos.kdd[idx.common.outcome,]
        varselected.kdd = read.csv(file=paste0(WORKING.DIR, "data/mix/rowwise/dat_batches/kdd_all_extended_20000fobs_multiclass_selecvars_alldataset.csv"), header=T, sep=",", stringsAsFactors = F)[1, -c(1,2)]
        kdd.norm        = 7
}


trials = cbind(train=c("nb15", "kdd", "ugr16", "kdd", "ugr16", "nb15"),
               test=c("ugr16", "ugr16", "nb15", "nb15", "kdd", "kdd"))

global <- NULL

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

# mean.x  <- colMeans(X.train)
# sd.x    <- apply(X.train, 2, sd)
# idx.0sd <- which(sd.x == 0)
# 
# X.train <- t(apply(X.train, 1, function(x) (x - mean.x) / sd.x))
# X.test  <- t(apply(X.test, 1, function(x) (x - mean.x) / sd.x))
# if (length(idx.0sd > 0)) {
#         X.train = X.train[,-idx.0sd]
#         X.test  = X.test[,-idx.0sd]
# }

cat("[", trials[i,1],"->", trials[i,2],"]:\n% clase en caja train: ",  paste0("(", paste0(names(table(y.train)), collapse = ","), ")"), " = ", table(y.train)/length(y.train), "\n% clase en caja test: ",  paste0("(", paste0(names(table(y.test)), collapse = ","), ")"), " = ", table(y.test)/length(y.test), "\n")

fitted.RF <- fit.RF(X = X.train, y = y.train)
predclass.test <- get.predictions(fitted.RF, X.test, type = "response")
probs.test     <- get.predictions(fitted.RF, X.test, type = "probs")

global <- rbind(global, c(trials[i,1], trials[i,2], compute.auc.global(y.test, probs.test), compute.acc.global(y.test, predclass.test)))

colnames(global) <- c("Train.dataset", "Test.dataset", "auc.multiclass", "auc.background", "auc.dos", "auc.scan", "auc.weighted", "acc.overall", "acc.background", "acc.dos", "acc.scan")


# global.res <- colMeans(global[,-c(1,2)])

write.table(matrix(global, nrow=1), file = OUTFILE, quote=F, col.names=F, row.names=F, append=TRUE, sep=",")
        
        