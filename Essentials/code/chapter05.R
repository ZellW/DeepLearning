## pdf("activation_functions.pdf", width = 3, height = 7)
## par(mfrow = c(3, 1))
## plot(x, x, type = "l", lwd = 2, ylab = "f(x) = x", main = "Linear")
## plot(x, tanh(x), type = "l", lwd = 2, ylab = "f(x) = tanh(x)", main = "Hyperbolic Tangent")
## plot(x, pmax(0, x), type = "l", lwd = 2, ylab = "f(x) = max(0, x)", main = "Rectifier")
## dev.off()

################################################################################
##                                                                            ##
##                                  Setup                                     ##
##                                                                            ##
################################################################################

source("checkpoint.R")
options(width = 70, digits = 2)

cl <- h2o.init(
  max_mem_size = "12G",
  nthreads = 4)


################################################################################
##                                                                            ##
##                         Picking the hyperparameters                        ##
##                                                                            ##
################################################################################

## data setup
digits.train <- read.csv("train.csv")
digits.train$label <- factor(digits.train$label, levels = 0:9)

h2odigits <- as.h2o(
  digits.train,
  destination_frame = "h2odigits")


i <- 1:32000
h2odigits.train <- h2odigits[i, ]

itest <- 32001:42000
h2odigits.test <- h2odigits[itest, ]
xnames <- colnames(h2odigits.train)[-1]


system.time(ex1 <- h2o.deeplearning(
  x = xnames,
  y = "label",
  training_frame= h2odigits.train,
  validation_frame = h2odigits.test,
  activation = "RectifierWithDropout",
  hidden = c(100),
  epochs = 10,
  adaptive_rate = FALSE,
  rate = .001,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(.2)
))

system.time(ex2 <- h2o.deeplearning(
  x = xnames,
  y = "label",
  training_frame= h2odigits.train,
  validation_frame = h2odigits.test,
  activation = "RectifierWithDropout",
  hidden = c(100),
  epochs = 10,
  adaptive_rate = FALSE,
  rate = .01,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(.2)
))


################################################################################
##                                                                            ##
##                      Training a Deep Prediction Model in R                 ##
##                                                                            ##
################################################################################


use.train.x <- read.table("UCI HAR Dataset/train/X_train.txt")
use.test.x <- read.table("UCI HAR Dataset/test/X_test.txt")

use.train.y <- read.table("UCI HAR Dataset/train/y_train.txt")[[1]]
use.test.y <- read.table("UCI HAR Dataset/test/y_test.txt")[[1]]

use.train <- cbind(use.train.x, Outcome = factor(use.train.y))
use.test <- cbind(use.test.x, Outcome = factor(use.test.y))

use.labels <- read.table("UCI HAR Dataset/activity_labels.txt")

h2oactivity.train <- as.h2o(
  use.train,
  destination_frame = "h2oactivitytrain")

h2oactivity.test <- as.h2o(
  use.test,
  destination_frame = "h2oactivitytest")

mt1 <- h2o.deeplearning(
  x = colnames(use.train.x),
  y = "Outcome",
  training_frame = h2oactivity.train,
  activation = "RectifierWithDropout",
  hidden = c(50),
  epochs = 10,
  rate = .005,
  loss = "CrossEntropy",
  input_dropout_ratio = .2,
  hidden_dropout_ratios = c(.5),
  export_weights_and_biases = TRUE
)

mt1



## deep features (hidden layer 1 neurons) as extracted
## using H2O functions
f <- as.data.frame(h2o.deepfeatures(mt1, h2oactivity.train, 1))
f[1:10, 1:5]

## weights for mapping from inputs to hidden layer 1 neurons
w1 <- as.matrix(h2o.weights(mt1, 1))

## plot heatmap of the weights
tmp <- as.data.frame(t(w1))
tmp$Row <- 1:nrow(tmp)
tmp <- melt(tmp, id.vars = c("Row"))

p.heat <- ggplot(tmp,
       aes(variable, Row, fill = value)) +
  geom_tile() +
  scale_fill_gradientn(colours = c("black", "white", "blue")) +
  theme_classic() +
  theme(axis.text = element_blank()) +
  xlab("Hidden Neuron") +
  ylab("Input Variable") +
  ggtitle("Heatmap of Weights for Layer 1")
print(p.heat)

png("../FirstDraft/chapter05_images/B4228_05_03.png",
    width = 5.5, height = 7.5, units = "in", res = 600)
print(p.heat)
dev.off()


###### manually score a model #######
## input data
d <- as.matrix(use.train[, -562])

## biases for hidden layer 1 neurons
b1 <- as.matrix(h2o.biases(mt1, 1))
## convert biases to a matrix with dimensions
## matching input data
b12 <- do.call(rbind, rep(list(t(b1)), nrow(d)))

#### Calculate features for layer 1

## step 1, scale input
d.scaled <- apply(d, 2, scale)

## step 2, apply weights using matrix multiplication
## and add the biases (which have replicated to be appropriate dimensions)
d.weighted <- d.scaled %*% t(w1) + b12

## step 3, weight to account for dropout
d.weighted <- d.weighted * (1 - .5)

## step 4, only use values > 0
d.weighted.rectifier <- apply(d.weighted, 2, pmax, 0)

all.equal(
  as.numeric(f[, 2]),
  d.weighted.rectifier[, 1],
  check.attributes = FALSE,
  use.names = FALSE,
  tolerance = 1e-04)



## weights for mapping from hidden layer 1 to outcome
w2 <- as.matrix(h2o.weights(mt1, 2))

## biases for outcome
b2 <- as.matrix(h2o.biases(mt1, 2))
b22 <- do.call(rbind, rep(list(t(b2)), nrow(d)))

yhat <- d.weighted.rectifier %*% t(w2) + b22

## softmax yhat
yhat <- exp(yhat)
normalizer <- do.call(cbind, rep(list(rowSums(yhat)), ncol(yhat)))
yhat <- yhat / normalizer


## pick column with maximum predicted probability
## and append as the "outcome" column
yhat <- cbind(Outcome = apply(yhat, 1, which.max), yhat)

yhat.h2o <- as.data.frame(h2o.predict(mt1, newdata = h2oactivity.train))

xtabs(~ yhat[, 1] + yhat.h2o[, 1])

################################################################################
##                                                                            ##
##                                 Use Case                                   ##
##                                                                            ##
################################################################################

## download and unzip the data (uncomment to run)
## download.file(
##   "http://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
##   destfile = "YearPredictionMSD.txt.zip")
## unzip("YearPredictionMSD.txt.zip")

## read data into R using fread() from data.table package
d <- fread("YearPredictionMSD.txt", sep = ",")

p.hist <- ggplot(d[, .(V1)], aes(V1)) +
  geom_histogram(binwidth = 1) +
  theme_classic() +
  xlab("Year of Release")
print(p.hist)

png("../FirstDraft/chapter05_images/B4228_05_04.png",
    width = 5.5, height = 5.5, units = "in", res = 600)
print(p.hist)
dev.off()

quantile(d$V1, probs = c(.005, .995))

d.train <- d[1:463715][V1 >= 1957 & V1 <= 2010]
d.test <- d[463716:515345][V1 >= 1957 & V1 <= 2010]

h2omsd.train <- as.h2o(
  d.train,
  destination_frame = "h2omsdtrain")
h2omsd.test <- as.h2o(
  d.test,
  destination_frame = "h2omsdtest")





summary(m0 <- lm(V1 ~ ., data = d.train))$r.squared

cor(
  d.test$V1,
  predict(m0, newdata = d.test))^2


system.time(m1 <- h2o.deeplearning(
  x = colnames(d)[-1],
  y = "V1",
  training_frame= h2omsd.train,
  validation_frame = h2omsd.test,
  activation = "RectifierWithDropout",
  hidden = c(50),
  epochs = 100,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  score_training_samples = 0,
  score_validation_samples = 0,
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
  )
)

m1



system.time(m2 <- h2o.deeplearning(
  x = colnames(d)[-1],
  y = "V1",
  training_frame= h2omsd.train,
  validation_frame = h2omsd.test,
  activation = "RectifierWithDropout",
  hidden = c(200, 200, 400),
  epochs = 100,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(.2, .2, .2),
  score_training_samples = 0,
  score_validation_samples = 0,
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
  )
)

m2


## system.time(m3 <- h2o.deeplearning(
##   x = colnames(d)[-1],
##   y = "V1",
##   training_frame= h2omsd.train,
##   validation_frame = h2omsd.test,
##   activation = "RectifierWithDropout",
##   hidden = c(500, 500, 1000),
##   epochs = 500,
##   input_dropout_ratio = 0,
##   hidden_dropout_ratios = c(.5, .5, .5),
##   score_training_samples = 0,
##   score_validation_samples = 0,
##   diagnostics = TRUE,
##   export_weights_and_biases = TRUE,
##   variable_importances = TRUE
##   )
## )
## m3


## m2b <- h2o.deeplearning(
##   x = colnames(d)[-1],
##   y = "V1",
##   training_frame= h2omsd.train,
##   validation_frame = h2omsd.test,
##   activation = "RectifierWithDropout",
##   hidden = c(200, 200, 400),
##   checkpoint = "DeepLearning_model_R_1452031055473_5",
##   epochs = 1000,
##   input_dropout_ratio = 0,
##   hidden_dropout_ratios = c(.2, .2, .2),
##   score_training_samples = 0,
##   score_validation_samples = 0,
##   diagnostics = TRUE,
##   export_weights_and_biases = TRUE,
##   variable_importances = TRUE
##   )
## m2b

############################
h2o.ls()

h2o.saveModel(
  object = m2,
  path = "c:\\Users\\jwile\\Google Drive\\Books\\DeepLearning\\R",
  force = TRUE)

## m2tmp <- h2o.loadModel("c:\\Users\\jwile\\Google Drive\\Books\\DeepLearning\\R\\DeepLearning_model_R_1452041883635_10")

## h2o.download_pojo()


#############################

h2o.scoreHistory(m2)


############################
## Residuals

yhat <- as.data.frame(h2o.predict(m2, h2omsd.train))
yhat <- cbind(as.data.frame(h2omsd.train[["V1"]]), yhat)

p.resid <- ggplot(yhat, aes(factor(V1), predict - V1)) +
  geom_boxplot() +
  geom_hline(yintercept = 0) +
  theme_classic() +
  theme(axis.text.x = element_text(
          angle = 90, vjust = 0.5, hjust = 0)) +
  xlab("Year of Release") +
  ylab("Residual (Predicted - Actual Year of Release)")
print(p.resid)


png("../FirstDraft/chapter05_images/B4228_05_05.png",
    width = 6, height = 6, units = "in", res = 600)
print(p.resid)
dev.off()


cor(yhat)^2
cor(subset(yhat, V1 <= 1980))^2
cor(subset(yhat, V1 > 1980))^2


############################

imp <- as.data.frame(h2o.varimp(m2))
imp[1:10, ]

p.imp <- ggplot(imp, aes(factor(variable, levels = variable), percentage)) +
  geom_point() +
  theme_classic() +
  theme(axis.text.x = element_blank()) +
  xlab("Variable Number") +
  ylab("Percentage of Total Importance")
print(p.imp)

png("../FirstDraft/chapter05_images/B4228_05_06.png",
    width = 4.5, height = 4.5, units = "in", res = 600)
print(p.imp)
dev.off()



mtest <- h2o.deeplearning(
  x = colnames(d)[2:13],
  y = "V1",
  training_frame= h2omsd.train,
  validation_frame = h2omsd.test,
  activation = "RectifierWithDropout",
  hidden = c(50),
  epochs = 100,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  score_training_samples = 0,
  score_validation_samples = 0,
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
)

mtest
