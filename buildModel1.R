#Look at https://datascienceplus.com/fitting-neural-network-in-r/
source("packages.R")

digits.train <- read.csv("./data/train.csv")
#dim(digits.train)
head(colnames(digits.train), 4)
tail(colnames(digits.train), 4)
head(digits.train[, 1:4])
unique(digits.train$label)

## convert to factor
digits.train$label <- factor(digits.train$label, levels = 0:9)
i <- 1:5000
digits.X <- digits.train[i, -1]
digits.y <- digits.train[i, 1]

barplot(table(digits.y))

set.seed(1234)
digits.m1 <- caret::train(x = digits.X, y = digits.y, method = "nnet", tuneGrid = expand.grid(.size = 5, .decay = 0.1),
          trControl = trainControl(method = "none"), MaxNWts = 10000, maxit = 100)

digits.yhat1 <- predict(digits.m1)
barplot(table(digits.yhat1))

digits.m2 <- train(digits.X, digits.y, method = "nnet", tuneGrid = expand.grid(.size = c(10), .decay = 0.1),
          trControl = trainControl(method = "none"), MaxNWts = 50000, maxit = 100)

digits.yhat2 <- predict(digits.m2)
barplot(table(digits.yhat2))

caret::confusionMatrix(xtabs(~digits.yhat2 + digits.y))

digits.m3 <- train(digits.X, digits.y, method = "nnet", tuneGrid = expand.grid(.size = c(40), .decay = 0.1),
                   trControl = trainControl(method = "none"), MaxNWts = 50000, maxit = 100)

digits.yhat3 <- predict(digits.m3)
barplot(table(digits.yhat3))

caret::confusionMatrix(xtabs(~digits.yhat3 + digits.y))

head(decodeClassLabels(digits.y))

digits.m4 <- mlp(as.matrix(digits.X), decodeClassLabels(digits.y), size = 40,learnFunc = "Rprop",
                 shufflePatterns = FALSE, maxit = 60)

digits.yhat4 <- fitted.values(digits.m4)
digits.yhat4 <- encodeClassLabels(digits.yhat4)
barplot(table(digits.yhat4))

caret::confusionMatrix(xtabs(~ I(digits.yhat4 - 1) + digits.y))

digits.yhat4.insample <- fitted.values(digits.m4)
head(round(digits.yhat4.insample, 2))

table(encodeClassLabels(digits.yhat4.insample, method = "WTA", l = 0, h = 0))
table(encodeClassLabels(digits.yhat4.insample, method = "WTA", l = 0, h = .5))
table(encodeClassLabels(digits.yhat4.insample, method = "WTA", l = .2, h = .5))
table(encodeClassLabels(digits.yhat4.insample, method = "402040", l = .4, h = .6))

i2 <- 5001:10000
digits.yhat4.pred <- predict(digits.m4, as.matrix(digits.train[i2, -1]))
table(encodeClassLabels(digits.yhat4.pred, method = "WTA", l = 0, h = 0))

caret::confusionMatrix(xtabs(~digits.train[i2, 1] + I(encodeClassLabels(digits.yhat4.pred) - 1)))

#UseCase
use.train.x <- read.table("./data/UseCase/train/X_train.txt")
use.train.y <- read.table("./data/UseCase/train/y_train.txt")[[1]]
use.test.x <- read.table("./data/UseCase/test/X_test.txt")
use.test.y <- read.table("./data/UseCase/test/y_test.txt")[[1]]
use.labels <- read.table("./data/UseCase/activity_labels.txt")

barplot(table(use.train.y))

tuning <- list(
     size = c(40, 20, 20, 50, 50),
     maxit = c(60, 100, 100, 100, 100),
     shuffle = c(FALSE, FALSE, TRUE, FALSE, FALSE),
     params = list(FALSE, FALSE, FALSE, FALSE, c(0.1, 20, 3)))

cl <- makeCluster(3)
clusterEvalQ(cl, {library(RSNNS)})
clusterExport(cl, c("tuning", "use.train.x", "use.train.y", "use.test.x", "use.test.y"))
registerDoSNOW(cl)

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

clusterExport(cl, "use.models")
use.yhat <- foreach(i = 1:5, .combine = 'c') %dopar% {
     list(list(Insample = encodeClassLabels(fitted.values(use.models[[i]])),
          Outsample = encodeClassLabels(predict(use.models[[i]], newdata = as.matrix(use.test.x)))
     ))
}

use.insample <- cbind(Y = use.train.y,
                      do.call(cbind.data.frame, lapply(use.yhat, `[[`, "Insample")))
colnames(use.insample) <- c("Y", paste0("Yhat", 1:5))
performance.insample <- do.call(rbind, lapply(1:5, function(i) {
     f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i))))
     use.dat <- use.insample[use.insample[,paste0("Yhat", i)] != 0, ]
     use.dat$Y <- factor(use.dat$Y, levels = 1:6)
     use.dat[, paste0("Yhat", i)] <- factor(use.dat[, paste0("Yhat", i)],
                                            levels = 1:6)
     res <- caret::confusionMatrix(xtabs(f, data = use.dat))
     cbind(Size = tuning$size[[i]],
           Maxit = tuning$maxit[[i]],
           Shuffle = tuning$shuffle[[i]],
           as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")])))
}))

performance.insample[,-4]

#Preventing Overfitting - p 47

set.seed(1234)
X <- mvrnorm(n = 200, mu = c(0, 0, 0, 0, 0),
             Sigma = matrix(c(
                  1, .9999, .99, .99, .10,
                  .9999, 1, .99, .99, .10,
                  .99, .99, 1, .99, .10,
                  .99, .99, .99, 1, .10,
                  .10, .10, .10, .10, 1
             ), ncol = 5))
y <- rnorm(200, 3 + X %*% matrix(c(1, 1, 1, 1, 0)), .5)

m.ols <- lm(y[1:100] ~ X[1:100, ])
m.lasso.cv <- cv.glmnet(X[1:100, ], y[1:100], alpha = 1)

plot(m.lasso.cv)

cbind(OLS = coef(m.ols), Lasso = coef(m.lasso.cv)[,1])

