################################################################################
##                                                                            ##
##                                  Setup                                     ##
##                                                                            ##
################################################################################


source("checkpoint.R")
options(width = 70, digits = 2)


################################################################################
##                                                                            ##
##               Building Neural Networks - Digit Recognition                 ##
##                                                                            ##
################################################################################

## https://www.kaggle.com/c/digit-recognizer

digits.train <- read.csv("train.csv")
dim(digits.train)
head(colnames(digits.train), 4)
tail(colnames(digits.train), 4)
head(digits.train[, 1:4])


## convert to factor
digits.train$label <- factor(digits.train$label, levels = 0:9)

i <- 1:5000
digits.X <- digits.train[i, -1]
digits.y <- digits.train[i, 1]

png("../FirstDraft/chapter02_images/B4228_02_01.png",
    width = 5, height = 5, units = "in", res = 600)
  barplot(table(digits.y))
dev.off()


set.seed(1234)
digits.m1 <- train(digits.X, digits.y,
           method = "nnet",
           tuneGrid = expand.grid(
             .size = c(5),
             .decay = 0.1),
           trControl = trainControl(method = "none"),
           MaxNWts = 10000,
           maxit = 100)

digits.yhat1 <- predict(digits.m1)

png("../FirstDraft/chapter02_images/B4228_02_02.png",
    width = 5, height = 5, units = "in", res = 600)
  barplot(table(digits.yhat1))
dev.off()

caret::confusionMatrix(xtabs(~digits.yhat1 + digits.y))

set.seed(1234)
digits.m2 <- train(digits.X, digits.y,
           method = "nnet",
           tuneGrid = expand.grid(
             .size = c(10),
             .decay = 0.1),
           trControl = trainControl(method = "none"),
            MaxNWts = 50000,
            maxit = 100)

digits.yhat2 <- predict(digits.m2)
png("../FirstDraft/chapter02_images/B4228_02_03.png",
    width = 5, height = 5, units = "in", res = 600)
  barplot(table(digits.yhat2))
dev.off()

caret::confusionMatrix(xtabs(~digits.yhat2 + digits.y))

set.seed(1234)
digits.m3 <- train(digits.X, digits.y,
           method = "nnet",
           tuneGrid = expand.grid(
             .size = c(40),
             .decay = 0.1),
           trControl = trainControl(method = "none"),
           MaxNWts = 50000,
           maxit = 100)

digits.yhat3 <- predict(digits.m3)
png("../FirstDraft/chapter02_images/B4228_02_04.png",
    width = 5, height = 5, units = "in", res = 600)
  barplot(table(digits.yhat3))
dev.off()

caret::confusionMatrix(xtabs(~digits.yhat3 + digits.y))

## using RSNNS package

head(decodeClassLabels(digits.y))

set.seed(1234)
digits.m4 <- mlp(as.matrix(digits.X),
             decodeClassLabels(digits.y),
             size = 40,
             learnFunc = "Rprop",
             shufflePatterns = FALSE,
             maxit = 60)

digits.yhat4 <- fitted.values(digits.m4)
digits.yhat4 <- encodeClassLabels(digits.yhat4)
png("../FirstDraft/chapter02_images/B4228_02_05.png",
    width = 5, height = 5, units = "in", res = 600)
  barplot(table(digits.yhat4))
dev.off()

caret::confusionMatrix(xtabs(~ I(digits.yhat4 - 1) + digits.y))


################################################################################
##                                                                            ##
##                        Predictions - Digit Recognition                     ##
##                                                                            ##
################################################################################

digits.yhat4.insample <- fitted.values(digits.m4)
head(round(digits.yhat4.insample, 2))

table(encodeClassLabels(digits.yhat4.insample,
                        method = "WTA", l = 0, h = 0))

table(encodeClassLabels(digits.yhat4.insample,
                        method = "WTA", l = 0, h = .5))

table(encodeClassLabels(digits.yhat4.insample,
                        method = "WTA", l = .2, h = .5))

table(encodeClassLabels(digits.yhat4.insample,
                        method = "402040", l = .4, h = .6))

i2 <- 5001:10000
digits.yhat4.pred <- predict(digits.m4,
                             as.matrix(digits.train[i2, -1]))

table(encodeClassLabels(digits.yhat4.pred,
                        method = "WTA", l = 0, h = 0))

################################################################################
##                                                                            ##
##                        Over Fitting - Digit Recognition                    ##
##                                                                            ##
################################################################################

caret::confusionMatrix(xtabs(~digits.train[i2, 1] +
  I(encodeClassLabels(digits.yhat4.pred) - 1)))



digits.yhat1.pred <- predict(digits.m1, digits.train[i2, -1])
digits.yhat2.pred <- predict(digits.m2, digits.train[i2, -1])
digits.yhat3.pred <- predict(digits.m3, digits.train[i2, -1])


measures <- c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")

n5.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat1))
n5.outsample <- caret::confusionMatrix(xtabs(~digits.train[i2, 1] + digits.yhat1.pred))

n10.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat2))
n10.outsample <- caret::confusionMatrix(xtabs(~digits.train[i2, 1] + digits.yhat2.pred))

n40.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat3))
n40.outsample <- caret::confusionMatrix(xtabs(~digits.train[i2, 1] + digits.yhat3.pred))

n40b.insample <- caret::confusionMatrix(xtabs(~digits.y + I(digits.yhat4 - 1)))
n40b.outsample <- caret::confusionMatrix(xtabs(~digits.train[i2, 1] +
  I(encodeClassLabels(digits.yhat4.pred) - 1)))


## results
shrinkage <- rbind(
  cbind(Size = 5, Sample = "In", as.data.frame(t(n5.insample$overall[measures]))),
  cbind(Size = 5, Sample = "Out", as.data.frame(t(n5.outsample$overall[measures]))),
  cbind(Size = 10, Sample = "In", as.data.frame(t(n10.insample$overall[measures]))),
  cbind(Size = 10, Sample = "Out", as.data.frame(t(n10.outsample$overall[measures]))),
  cbind(Size = 40, Sample = "In", as.data.frame(t(n40.insample$overall[measures]))),
  cbind(Size = 40, Sample = "Out", as.data.frame(t(n40.outsample$overall[measures]))),
  cbind(Size = 40, Sample = "In", as.data.frame(t(n40b.insample$overall[measures]))),
  cbind(Size = 40, Sample = "Out", as.data.frame(t(n40b.outsample$overall[measures])))
  )
shrinkage$Pkg <- rep(c("nnet", "RSNNS"), c(6, 2))

dodge <- position_dodge(width=0.4)

p.shrinkage <- ggplot(shrinkage, aes(interaction(Size, Pkg, sep = " : "), Accuracy,
                      ymin = AccuracyLower, ymax = AccuracyUpper,
                      shape = Sample, linetype = Sample)) +
  geom_point(size = 2.5, position = dodge) +
  geom_errorbar(width = .25, position = dodge) +
  xlab("") + ylab("Accuracy + 95% CI") +
  theme_classic() +
  theme(legend.key.size = unit(1, "cm"), legend.position = c(.8, .2))

png("../FirstDraft/chapter02_images/B4228_02_06.png",
    width = 6, height = 6, units = "in", res = 600)
  print(p.shrinkage)
dev.off()


################################################################################
##                                                                            ##
##                                   Use Case                                 ##
##                                                                            ##
################################################################################


## http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

## Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

use.train.x <- read.table("UCI HAR Dataset/train/X_train.txt")
use.train.y <- read.table("UCI HAR Dataset/train/y_train.txt")[[1]]

use.test.x <- read.table("UCI HAR Dataset/test/X_test.txt")
use.test.y <- read.table("UCI HAR Dataset/test/y_test.txt")[[1]]

use.labels <- read.table("UCI HAR Dataset/activity_labels.txt")

png("../FirstDraft/chapter02_images/B4228_02_07.png",
    width = 6, height = 6, units = "in", res = 600)
  barplot(table(use.train.y))
dev.off()


## choose tuning parameters
tuning <- list(
  size = c(40, 20, 20, 50, 50),
  maxit = c(60, 100, 100, 100, 100),
  shuffle = c(FALSE, FALSE, TRUE, FALSE, FALSE),
  params = list(FALSE, FALSE, FALSE, FALSE, c(0.1, 20, 3)))

## setup cluster using 5 cores
## load packages, export required data and variables
## and register as a backend for use with the foreach package
cl <- makeCluster(5)
clusterEvalQ(cl, {
  source("checkpoint.R")
})
clusterExport(cl,
  c("tuning", "use.train.x", "use.train.y",
    "use.test.x", "use.test.y")
  )
registerDoSNOW(cl)

## train models in parallel
use.models <- foreach(i = 1:5, .combine = 'c') %dopar% {
  if (tuning$params[[i]][1]) {
    set.seed(1234)
    list(Model = mlp(
      as.matrix(use.train.x),
      decodeClassLabels(use.train.y),
      size = tuning$size[[i]],
      learnFunc = "Rprop",
      shufflePatterns = tuning$shuffle[[i]],
      learnFuncParams = tuning$params[[i]],
      maxit = tuning$maxit[[i]]
      ))
  } else {
    set.seed(1234)
    list(Model = mlp(
      as.matrix(use.train.x),
      decodeClassLabels(use.train.y),
      size = tuning$size[[i]],
      learnFunc = "Rprop",
      shufflePatterns = tuning$shuffle[[i]],
      maxit = tuning$maxit[[i]]
    ))
  }
}

## export models and calculate both in sample,
## 'fitted' and out of sample 'predicted' values
clusterExport(cl, "use.models")
use.yhat <- foreach(i = 1:5, .combine = 'c') %dopar% {
  list(list(
    Insample = encodeClassLabels(fitted.values(use.models[[i]])),
    Outsample = encodeClassLabels(predict(use.models[[i]],
                                          newdata = as.matrix(use.test.x)))
    ))
}

use.insample <- cbind(Y = use.train.y,
  do.call(cbind.data.frame, lapply(use.yhat, `[[`, "Insample")))
colnames(use.insample) <- c("Y", paste0("Yhat", 1:5))

performance.insample <- do.call(rbind, lapply(1:5, function(i) {
  f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i))))
  use.dat <- use.insample[use.insample[,paste0("Yhat", i)] != 0, ]
  use.dat$Y <- factor(use.dat$Y, levels = 1:6)
  use.dat[, paste0("Yhat", i)] <- factor(use.dat[, paste0("Yhat", i)], levels = 1:6)
  res <- caret::confusionMatrix(xtabs(f, data = use.dat))

  cbind(Size = tuning$size[[i]],
        Maxit = tuning$maxit[[i]],
        Shuffle = tuning$shuffle[[i]],
        as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")])))
}))

use.outsample <- cbind(Y = use.test.y,
  do.call(cbind.data.frame, lapply(use.yhat, `[[`, "Outsample")))
colnames(use.outsample) <- c("Y", paste0("Yhat", 1:5))
performance.outsample <- do.call(rbind, lapply(1:5, function(i) {
  f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i))))
  use.dat <- use.outsample[use.outsample[,paste0("Yhat", i)] != 0, ]
  use.dat$Y <- factor(use.dat$Y, levels = 1:6)
  use.dat[, paste0("Yhat", i)] <- factor(use.dat[, paste0("Yhat", i)], levels = 1:6)
  res <- caret::confusionMatrix(xtabs(f, data = use.dat))

  cbind(Size = tuning$size[[i]],
        Maxit = tuning$maxit[[i]],
        Shuffle = tuning$shuffle[[i]],
        as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")])))
}))


options(width = 80)
performance.insample[,-4]

performance.outsample[,-4]

