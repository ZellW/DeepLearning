setwd("~/GitHub/DeepLearning")

# pkgs <- c("keras", "lime", "tidyquant", "rsample", "recipes", "yardstick", "corrr")
# install.packages(pkgs)

library(keras)
library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)
library(readr)

churn_data_raw <- read_csv("../CaseStudyChurnSurvival/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

glimpse(churn_data_raw)

churn_data_tbl <- churn_data_raw %>% select(-customerID) %>% drop_na() %>% select(Churn, everything())

glimpse(churn_data_tbl)

set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop = 0.8)
train_test_split

train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split) 

# Determine if log transformation improves correlation 
# between TotalCharges and Churn
train_tbl %>% select(Churn, TotalCharges) %>% mutate(Churn = Churn %>% as.factor() %>% as.numeric(),
                                                     LogTotalCharges = log(TotalCharges)) %>%
      corrr:correlate() %>% corrr::focus(Churn) %>% corrr::fashion()
# gets the correlation, selects only the Churn variable and formats it. (Simply using corrr functions over dplyr)
# log transform is more correlated so it will be used


rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
      step_discretize(tenure, options = list(cuts = 6)) %>%
      step_log(TotalCharges) %>%
      step_dummy(all_nominal(), -all_outcomes()) %>%
      step_center(all_predictors(), -all_outcomes()) %>%
      step_scale(all_predictors(), -all_outcomes()) %>%
      prep(data = train_tbl)


x_train_tbl <- bake(rec_obj, new_data = train_tbl)
x_test_tbl  <- bake(rec_obj, new_data = test_tbl)

glimpse(x_train_tbl)

# Response variables for training and testing sets
y_train_vec <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test_vec  <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)

# Building our Artificial Neural Network
model_keras <- keras_model_sequential()

model_keras %>% 
      # First hidden layer
      layer_dense(
            units              = 16, 
            kernel_initializer = "uniform", 
            activation         = "relu", 
            input_shape        = ncol(x_train_tbl)) %>% 
      # Dropout to prevent overfitting
      layer_dropout(rate = 0.1) %>%
      # Second hidden layer
      layer_dense(
            units              = 16, 
            kernel_initializer = "uniform", 
            activation         = "relu") %>% 
      # Dropout to prevent overfitting
      layer_dropout(rate = 0.1) %>%
      # Output layer
      layer_dense(
            units              = 1, 
            kernel_initializer = "uniform", 
            activation         = "sigmoid") %>% 
      # Compile ANN
      compile(
            optimizer = 'adam',
            loss      = 'binary_crossentropy',
            metrics   = c('accuracy')
      )
model_keras
