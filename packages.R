## uncomment to install the checkpoint package
## install.packages("checkpoint")
# library(checkpoint)
# 
# checkpoint("2016-02-20", R.version = "3.2.4")#Cant seem to get it to work
# 
#library(readr)#Cannot use.  “The data should not be a ‘tbl_df’ or ‘tbl’.  
#A likely possibility is the reader loaded and used the dplyr package for some data management 
#in addition to the suggested packages. Specifically tbl_df does not behave the same way as 
#regular data frames in R, so for the example below, digits.y is a vector when using regular 
#data frames (the dimensions of the data frame are dropped) but in tbl_df it would never simplify 
#(drop the dimensions) so digits.y would still be a data frame, and that can throw off train() later on.

## Chapter 1 ##

## Tools
library(RCurl)
library(jsonlite)
library(caret)
library(e1071)

## basic stats packages
library(statmod)
library(MASS)

## neural networks
library(nnet)
library(neuralnet)
library(RSNNS)

## deep learning
library(deepnet)
library(darch)
#Notes for h20
#cl <- h2o.init(max_mem_size = "3G", nthreads = 2)
# 
# H2O is not running yet, starting it now...
# Error in system2(command, "-version", stdout = TRUE, stderr = TRUE) : 
#      '""' not found
# You need to install the JDK and point the JAVA_HOME environment variable 
# to the JDK directory (the parent of the bin directory), if it isn't 
# automatically done by the installer.
# (http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
library(h2o)



## Chapter 2 ##
library(parallel)
library(foreach)
library(doSNOW)

## Chapter 3 ##
library(glmnet)

## Chapter 4 ##
library(data.table)

## Chapter 5 ##

## Chapter 6 ##
library(gridExtra)
library(mgcv)

cl <- h2o.init(max_mem_size = "3G", nthreads = 2)
#http://127.0.0.1:54321/
