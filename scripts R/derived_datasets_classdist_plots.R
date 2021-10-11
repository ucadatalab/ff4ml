rm(list=ls())

library(ggplot2)
library(plyr)



load.ugr16 <- function(n) {
        if (n == 10000) {
                datos.ugr = read.csv(file=paste0("data/ugr16/dat_batches/output-UGR16_all_extended_10000fobs_395082bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        } else {
                datos.ugr = read.csv(file=paste0("data/ugr16/dat_batches/output-UGR16_all_extended_20000fobs_197541bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        }
        datos.ugr$outcome   <- gsub("neris", "", gsub("anomaly_", "", gsub("label_", "", datos.ugr$outcome)))
        idx.toremove        <- which(datos.ugr$outcome %in% c("sshscan","udpscan"))
        datos.ugr           <- datos.ugr[-idx.toremove, "outcome", drop=F]
        datos.ugr$outcome   <- gsub(pattern = "background", replacement = "Background", x = datos.ugr$outcome)
        datos.ugr$outcome   <- gsub(pattern = "botnet", replacement = "Botnet", x = datos.ugr$outcome)
        datos.ugr$outcome   <- gsub(pattern = "dos", replacement = "DoS", x = datos.ugr$outcome)
        datos.ugr$outcome   <- gsub(pattern = "scan", replacement = "Port Scanning", x = datos.ugr$outcome)
        datos.ugr$outcome   <- gsub(pattern = "spam", replacement = "Spam", x = datos.ugr$outcome)
        datos.ugr$outcome   <- as.factor(datos.ugr$outcome)
        colnames(datos.ugr) <- "Class"
        
        datos.ugr
}


load.nb15 <- function(n) {
        if (n == 10000) {
                datos.nb15 = read.csv(file=paste0("data/unsw-nb15/dat_batches/output-NB15_all_extended_10000fobs_254bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        } else {
                datos.nb15       = read.csv(file=paste0("data/unsw-nb15/dat_batches/output-NB15_all_extended_20000fobs_127bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        }
        datos.nb15$outcome <- gsub("label_", "", datos.nb15$outcome)
        idx.toremove       <- which(datos.nb15$outcome %in% c("analysis","backdoor","shellcode"))
        datos.nb15         <- datos.nb15[-idx.toremove, "outcome", drop=F]
        datos.nb15$outcome <- gsub(pattern = "background", replacement = "Background", x = datos.nb15$outcome)
        datos.nb15$outcome <- gsub(pattern = "dos", replacement = "DoS", x = datos.nb15$outcome)
        datos.nb15$outcome <- gsub(pattern = "exploit", replacement = "Exploit", x = datos.nb15$outcome)
        datos.nb15$outcome <- gsub(pattern = "fuzzer", replacement = "Fuzzer", x = datos.nb15$outcome)
        datos.nb15$outcome <- gsub(pattern = "generic", replacement = "Generic", x = datos.nb15$outcome)
        datos.nb15$outcome <- gsub(pattern = "reconnaissance", replacement = "Port Scanning", x = datos.nb15$outcome)
        datos.nb15$outcome <- as.factor(datos.nb15$outcome)
        colnames(datos.nb15) <- "Class"
        
        datos.nb15
}


load.kdd <- function(n) {
        if (n == 10000) {
                datos.kdd       = read.csv(file=paste0("data/nsl-kdd/dat_batches/output-KDD_all_extended_10000fobs_14bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        } else {
                datos.kdd       = read.csv(file=paste0("data/nsl-kdd/dat_batches/output-KDD_all_extended_20000fobs_7bsize_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        }
        datos.kdd$outcome <- gsub("label_category_", "", datos.kdd$outcome)
        idx.toremove      <- which(datos.kdd$outcome %in% c("u2r"))
        datos.kdd         <- datos.kdd[-idx.toremove, "outcome", drop=F]
        datos.kdd$outcome <- gsub(pattern = "background", replacement = "Background", x = datos.kdd$outcome)
        datos.kdd$outcome <- gsub(pattern = "dos", replacement = "DoS", x = datos.kdd$outcome)
        datos.kdd$outcome <- gsub(pattern = "probe", replacement = "Port Scanning", x = datos.kdd$outcome)
        datos.kdd$outcome <- gsub(pattern = "r2l", replacement = "R2L", x = datos.kdd$outcome)
        datos.kdd$outcome <- as.factor(datos.kdd$outcome)
        colnames(datos.kdd) <- "Class"
        
        datos.kdd
}


load.mix <- function(n) {
        datos.mix = read.csv(file=paste0("data/mix/rowwise/dat_batches/output-rowise_", n, "fobs_multiclass.csv"), header=T, sep=",", stringsAsFactors = F)
        datos.mix         <- datos.mix[, "outcome", drop=F]
        datos.mix$outcome <- gsub("background", "Background", datos.mix$outcome)
        datos.mix$outcome <- gsub("dos", "DoS", datos.mix$outcome)
        datos.mix$outcome <- gsub("scan", "Port Scanning", datos.mix$outcome)
        datos.mix$outcome <- as.factor(datos.mix$outcome)
        colnames(datos.mix) <- "Class"
        
        datos.mix
}



setwd("/.../.../.../.../ff4ml/")


# 1. UGR'16
ugr16.10k <- load.ugr16(n=10000)
ugr16.20k <- load.ugr16(n=20000)
ugr16.plot.data <- rbind(data.frame(N=rep(nrow(ugr16.10k), length(levels(ugr16.10k$Class))), Class=levels(ugr16.10k$Class), Percentage=as.numeric(round(table(ugr16.10k)/nrow(ugr16.10k)*100,2))),
                         data.frame(N=rep(nrow(ugr16.20k), length(levels(ugr16.20k$Class))), Class=levels(ugr16.20k$Class), Percentage=as.numeric(round(table(ugr16.20k)/nrow(ugr16.20k)*100,2))))
ugr16.plot.data$N <- as.factor(ugr16.plot.data$N)

my_colors = c("#aa381e", "#00b0ff", "#3b85ae","#466da5","#263fc9")#,"#1f3297")
p1 <- ugr16.plot.data %>% ggplot(aes(x=N, y=Percentage, fill=Class)) +
        geom_bar(width=0.7, position = position_dodge(width = 0.8), stat = "identity") +
        scale_fill_manual(values=my_colors) +
        ylim(0,59) + 
        geom_text(aes(label=Percentage), position = position_dodge2(width=0.8, preserve = "single"), vjust=-0.3, size=3) +
        labs(x = "Total samples on each derived dataset", y = "Class distribution (%)") +
        theme_bw()
# save as PDF 6 x 4 inches


# 2. UNSW-NB15
nb15.10k <- load.nb15(n=10000)
nb15.20k <- load.nb15(n=20000)
nb15.plot.data <- rbind(data.frame(N=rep(nrow(nb15.10k), length(levels(nb15.10k$Class))), Class=levels(nb15.10k$Class), Percentage=as.numeric(round(table(nb15.10k)/nrow(nb15.10k)*100,2))),
                         data.frame(N=rep(nrow(nb15.20k), length(levels(nb15.20k$Class))), Class=levels(nb15.20k$Class), Percentage=as.numeric(round(table(nb15.20k)/nrow(nb15.20k)*100,2))))
nb15.plot.data$N <- as.factor(nb15.plot.data$N)

my_colors = c("#aa381e", "#3b85ae", "#00b0ff", "#263fc9","#00007c", "#466da5")#,"#1f3297")
p2 <- nb15.plot.data %>% ggplot(aes(x=N, y=Percentage, fill=Class)) +
        geom_bar(width=0.7, position = position_dodge(width = 0.8), stat = "identity") +
        scale_fill_manual(values=my_colors) +
        ylim(0,59) + 
        geom_text(aes(label=Percentage), position = position_dodge2(width=0.8, preserve = "single"), vjust=-0.3, size=3) +
        labs(x = "Total samples on each derived dataset", y = "Class distribution (%)") +
        theme_bw()
# save as PDF 6 x 4 inches


# 3. NSL-KDD
kdd.10k <- load.kdd(n=10000)
kdd.20k <- load.kdd(n=20000)
kdd.plot.data <- rbind(data.frame(N=rep(nrow(kdd.10k), length(levels(kdd.10k$Class))), Class=levels(kdd.10k$Class), Percentage=as.numeric(round(table(kdd.10k)/nrow(kdd.10k)*100,2))),
                        data.frame(N=rep(nrow(kdd.20k), length(levels(kdd.20k$Class))), Class=levels(kdd.20k$Class), Percentage=as.numeric(round(table(kdd.20k)/nrow(kdd.20k)*100,2))))
kdd.plot.data$N <- as.factor(kdd.plot.data$N)

my_colors = c("#aa381e", "#3b85ae", "#466da5", "#263fc9")
p3 <- kdd.plot.data %>% ggplot(aes(x=N, y=Percentage, fill=Class)) +
        geom_bar(width=0.7, position = position_dodge(width = 0.8), stat = "identity") +
        scale_fill_manual(values=my_colors) +
        ylim(0,91) + 
        geom_text(aes(label=Percentage), position = position_dodge2(width=0.8, preserve = "single"), vjust=-0.3, size=3) +
        labs(x = "Total samples on each derived dataset", y = "Class distribution (%)") +
        theme_bw()
# save as PDF 6 x 4 inches


# 4. Mix
mix.10k <- load.mix(n=10000)
mix.20k <- load.mix(n=20000)
mix.plot.data <- rbind(data.frame(N=rep(nrow(mix.10k), length(levels(mix.10k$Class))), Class=levels(mix.10k$Class), Percentage=as.numeric(round(table(mix.10k)/nrow(mix.10k)*100,2))),
                       data.frame(N=rep(nrow(mix.20k), length(levels(mix.20k$Class))), Class=levels(mix.20k$Class), Percentage=as.numeric(round(table(mix.20k)/nrow(mix.20k)*100,2))))
mix.plot.data$N <- as.factor(mix.plot.data$N)

my_colors = c("#aa381e", "#3b85ae", "#466da5")
p4 <- mix.plot.data %>% ggplot(aes(x=N, y=Percentage, fill=Class)) +
        geom_bar(width=0.7, position = position_dodge(width = 0.8), stat = "identity") +
        scale_fill_manual(values=my_colors) +
        ylim(0,91) + 
        geom_text(aes(label=Percentage), position = position_dodge2(width=0.8, preserve = "single"), vjust=-0.3, size=3) +
        labs(x = "Total samples on each derived dataset", y = "Class distribution (%)") +
        theme_bw()
# save as PDF 6 x 4 inches

