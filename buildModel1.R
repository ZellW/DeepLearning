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
