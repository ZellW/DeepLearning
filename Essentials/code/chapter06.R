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
##                          Dealing with missing data                         ##
##                                                                            ##
################################################################################

## setup iris data with some missing
d <- as.data.table(iris)
d[Species == "setosa", c("Petal.Width", "Petal.Length") := .(NA, NA)]

h2o.dmiss <- as.h2o(d, destination_frame="iris_missing")
h2o.dmeanimp <- as.h2o(d, destination_frame="iris_missing_imp")

## mean imputation
missing.cols <- colnames(h2o.dmiss)[apply(d, 2, anyNA)]

for (v in missing.cols) {
  h2o.dmeanimp <- h2o.impute(h2o.dmeanimp, column = v)
}

## random forest imputation
d.imputed <- d

## prediction model
for (v in missing.cols) {
  tmp.m <- h2o.randomForest(
    x = setdiff(colnames(h2o.dmiss), v),
    y = v,
    training_frame = h2o.dmiss)
  yhat <- as.data.frame(h2o.predict(tmp.m, newdata = h2o.dmiss))
  d.imputed[[v]] <- ifelse(is.na(d.imputed[[v]]), yhat$predict, d.imputed[[v]])
}


png("../FirstDraft/chapter06_images/B4228_06_01.png",
    width = 5, height = 12, units = "in", res = 600)

grid.arrange(
  ggplot(iris, aes(Petal.Length, Petal.Width,
    colour = Species, shape = Species)) +
    geom_point() +
    theme_classic() +
    ggtitle("Original Data"),
 ggplot(as.data.frame(h2o.dmeanimp), aes(Petal.Length, Petal.Width,
    colour = Species, shape = Species)) +
    geom_point() +
    theme_classic() +
   ggtitle("Mean Imputed Data"),
 ggplot(d.imputed, aes(Petal.Length, Petal.Width,
    colour = Species, shape = Species)) +
    geom_point() +
    theme_classic() +
   ggtitle("Random Forest Imputed Data"),
  ncol = 1)

dev.off()




################################################################################
##                                                                            ##
##                          Find best hyperparameters                         ##
##                                                                            ##
################################################################################

expand.grid(
  layers = c(1, 2, 4),
  epochs = c(50, 100),
  l1 = c(.001, .01, .05))

## plot of the density of the beta
png("../FirstDraft/chapter06_images/B4228_06_02.png",
    width = 4, height = 8, units = "in", res = 600)

par(mfrow = c(2, 1))
plot(
  seq(0, .5, by = .001),
  dbeta(seq(0, .5, by = .001), 1, 12),
  type = "l", xlab = "x", ylab = "Density",
  main = "Density of a beta(1, 12)")

plot(
  seq(0, 1, by = .001)/2,
  dbeta(seq(0, 1, by = .001), 1.5, 1),
  type = "l", xlab = "x", ylab = "Density",
  main = "Density of a beta(1.5, 1) / 2")

dev.off()

run <- function(seed, name = paste0("m_", seed), run = TRUE) {
  set.seed(seed)

  p <- list(
    Name = name,
    seed = seed,
    depth = sample(1:5, 1),
    l1 = runif(1, 0, .01),
    l2 = runif(1, 0, .01),
    input_dropout = rbeta(1, 1, 12),
    rho = runif(1, .9, .999),
    epsilon = runif(1, 1e-10, 1e-4))

  p$neurons <- sample(20:600, p$depth, TRUE)
  p$hidden_dropout <- rbeta(p$depth, 1.5, 1)/2

  if (run) {
  model <- h2o.deeplearning(
    x = colnames(use.train.x),
    y = "Outcome",
    training_frame = h2oactivity.train,
    activation = "RectifierWithDropout",
    hidden = p$neurons,
    epochs = 100,
    loss = "CrossEntropy",
    input_dropout_ratio = p$input_dropout,
    hidden_dropout_ratios = p$hidden_dropout,
    l1 = p$l1,
    l2 = p$l2,
    rho = p$rho,
    epsilon = p$epsilon,
    export_weights_and_biases = TRUE,
    model_id = p$Name
  )

  ## performance on training data
  p$MSE <- h2o.mse(model)
  p$R2 <- h2o.r2(model)
  p$Logloss <- h2o.logloss(model)
  p$CM <- h2o.confusionMatrix(model)

  ## performance on testing data
  perf <- h2o.performance(model, h2oactivity.test)
  p$T.MSE <- h2o.mse(perf)
  p$T.R2 <- h2o.r2(perf)
  p$T.Logloss <- h2o.logloss(perf)
  p$T.CM <- h2o.confusionMatrix(perf)

  } else {
    model <- NULL
  }

  return(list(
    Params = p,
    Model = model))
}





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



use.seeds <- c(403L, 10L, 329737957L, -753102721L, 1148078598L, -1945176688L,
-1395587021L, -1662228527L, 367521152L, 217718878L, 1370247081L,
571790939L, -2065569174L, 1584125708L, 1987682639L, 818264581L,
1748945084L, 264331666L, 1408989837L, 2010310855L, 1080941998L,
1107560456L, -1697965045L, 1540094185L, 1807685560L, 2015326310L,
-1685044991L, 1348376467L, -1013192638L, -757809164L, 1815878135L,
-1183855123L, -91578748L, -1942404950L, -846262763L, -497569105L,
-1489909578L, 1992656608L, -778110429L, -313088703L, -758818768L,
-696909234L, 673359545L, 1084007115L, -1140731014L, -877493636L,
-1319881025L, 3030933L, -154241108L, -1831664254L)


model.res <- lapply(use.seeds, run)


model.res.dat <- do.call(rbind, lapply(model.res, function(x) with(x$Params,
  data.frame(l1 = l1, l2 = l2,
             depth = depth, input_dropout = input_dropout,
             SumNeurons = sum(neurons),
             MeanHiddenDropout = mean(hidden_dropout),
             rho = rho, epsilon = epsilon, MSE = T.MSE))))


p.perf <- ggplot(melt(model.res.dat, id.vars = c("MSE")), aes(value, MSE)) +
  geom_point() +
  stat_smooth(colour = "black") +
  facet_wrap(~ variable, scales = "free_x", ncol = 2) +
  theme_classic()
print(p.perf)

png("../FirstDraft/chapter06_images/B4228_06_03.png",
    width = 6, height = 12, units = "in", res = 600)
print(p.perf)
dev.off()


summary(m.gam <- gam(MSE ~ s(l1, k = 4) +
              s(l2, k = 4) +
              s(input_dropout) +
              s(rho, k = 4) +
              s(epsilon, k = 4) +
              s(MeanHiddenDropout, k = 4) +
              te(depth, SumNeurons, k = 4),
            data = model.res.dat))


png("../FirstDraft/chapter06_images/B4228_06_04.png",
    width = 5.5, height = 9, units = "in", res = 600)

par(mfrow = c(3, 2))
for (i in 1:6) {
  plot(m.gam, select = i)
}

dev.off()


png("../FirstDraft/chapter06_images/B4228_06_05.png",
    width = 5, height = 6.5, units = "in", res = 600)

plot(m.gam, select = 7)

dev.off()




model.optimized <- h2o.deeplearning(
    x = colnames(use.train.x),
    y = "Outcome",
    training_frame = h2oactivity.train,
    activation = "RectifierWithDropout",
    hidden = c(500, 500, 500),
    epochs = 100,
    loss = "CrossEntropy",
    input_dropout_ratio = .08,
    hidden_dropout_ratios = c(.50, .50, .50),
    l1 = .002,
    l2 = 0,
    rho = .95,
    epsilon = 1e-10,
    export_weights_and_biases = TRUE,
    model_id = "optimized_model"
)

h2o.performance(model.optimized, h2oactivity.test)

model.res.dat[which.min(model.res.dat$MSE), ]

