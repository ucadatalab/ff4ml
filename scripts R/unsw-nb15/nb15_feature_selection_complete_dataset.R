rm(list=ls())

library(caret)
library(glmnet)


NUMOBS   = "20000"

dataset = read.csv(file=paste0("./data/unsw-nb15/dat_batches/output-nb15_all_extended_", NUMOBS, "fobs_multiclass.csv"), header=T, sep=",")
outfile.name = paste0("./data/unsw-nb15/dat_batches/nb15_all_extended_", NUMOBS, "fobs_multiclass_selecvars_alldataset.csv")
X = dataset[,1:134]
y = dataset[,ncol(dataset)]

header = colnames(X)
write.table(t(as.matrix(header)), file = outfile.name, sep = ",", append = F, quote = F, row.names = F, col.names = F)

set.seed(1234)


X.train = X
y.train = y
names.few.obs = names(which(table(y.train)<12))
idx.few.obs = which(y.train %in% names.few.obs)
if (length(idx.few.obs) > 0) {
        X.train = X.train[-idx.few.obs,]
        y.train = y.train[-idx.few.obs]
}
y.train = factor(as.character(y.train))

inner.folds = createFolds(y = y.train, k=3, list=F)
weights     = rep(nrow(X.train), nrow(X.train))
for (clase in levels(y.train)) {
        idx.clase = which(y.train == clase)
        weights[idx.clase] = weights[idx.clase]-length(idx.clase)
}

lasso.model = cv.glmnet(x = as.matrix(X.train), y = y.train, weights = weights,
                        family = "multinomial", type.multinomial = "grouped",
                        alpha = 1, foldid = inner.folds, parallel = TRUE)

lasso.coefs = coef(lasso.model)
varsnames   = rownames(lasso.coefs[[1]])
lasso.coefs = sapply(lasso.coefs, function(x) as.matrix(x))
rownames(lasso.coefs) = varsnames
lasso.coefs = lasso.coefs[-1,]
idx.retained = which(apply(lasso.coefs, 1, function(x) all(x!=0)))

selected.variables = rownames(lasso.coefs)[idx.retained]
cat("Seleccionadas", paste0(length(selected.variables), "/", ncol(X.train)), "\n")

row.vars.selected = rep(0, ncol(X.train))
names(row.vars.selected) = colnames(X.train)
row.vars.selected[selected.variables] = 1

to.save = row.vars.selected
write.table(t(as.matrix(to.save)), file = outfile.name, sep = ",", append = T, quote = F, row.names = F, col.names = F)