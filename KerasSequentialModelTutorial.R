##############################################
## Keras sequential model tutorial 
## Repo: https://github.com/pablo14/Keras-R-tutorials/blob/master/keras_sequential_model.R
## Post explaining everything on this script: https://blog.datascienceheroes.com/how-to-create-a-sequential-model-in-keras-for-r/
## Pablo Casas | @pabloc_ds
##############################################

## Installing / Loading Keras
library(reticulate)
library(tensorflow)
# install.packages("keras")
library(keras)
library(tidyverse)

# Input: 10000 rows and 3 columns of uniform distribution
x_data = matrix(data = runif(30000), nrow = 10000, ncol = 3)

# Output
y_data = ifelse(rowSums(x_data) > 1.5, 1, 0)

y_data_oneh = to_categorical(y_data, num_classes = 2)

head(y_data_oneh)

## Creating the sequential model
model = keras_model_sequential() %>%   
      layer_dense(units = 64, activation = "relu", input_shape = ncol(x_data)) %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = ncol(y_data_oneh), activation = "softmax")

model

compile(model, loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(), metrics = "accuracy")

history = fit(model,  x_data, y_data_oneh, epochs = 20, batch_size = 128, validation_split = 0.2)

plot(history)


## Validating with unseen data

x_data_test = matrix(data = runif(3000), nrow = 1000, ncol = 3)
dim(x_data_test) 

y_data_pred = predict_classes(model, x_data_test)

glimpse(y_data_pred)

y_data_pred_oneh = predict(model, x_data_test)

dim(y_data_pred_oneh)
head(y_data_pred_oneh)

y_data_real = ifelse(rowSums(x_data_test) > 1.5, 1, 0)
y_data_real_oneh = to_categorical(y_data_real)

## Evaluation on training data
evaluate(model, x_data, y_data_oneh, verbose = 0)

## Evaluation on Test data (we need the one-hot version)
evaluate(model, x_data_test, y_data_real_oneh, verbose = 0)
