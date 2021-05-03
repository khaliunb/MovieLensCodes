#### RMSE calculation script for MovieLens project
#### This file holds the script that is based on the edx data set and will not use final validation data set
#### Created by Khaliun.B 2021.04.11
#### According to the MovieLens project instructions
#### (!) Second, you will train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set.
#### 25 points: RMSE < 0.86490

#### (!) You should split the edx data into separate training and test sets to design and test your algorithm.

#### (!) The validation data (the final hold-out test set) should NOT be used for training, developing, or selecting your algorithm and it should ONLY be used for evaluating the RMSE of your final algorithm.

source("MovieLensAnalysisScripts.R", local = knitr::knit_global())

###########################################################################
### BEGIN: This chunk of code is training Linear model with Regularized Movie Effect + User Effect model
### on edx data set and compares the prediction results with validation data set to calculate final RMSE.
### (!) We are using lamdbda value of 4.75 gained from analysis of edx data
### For complete tuning process of lambda value please refer to code file MovieLensAnalysisScript_DataExploration.R
### Analysis results are also included in Final Report. Please refer to file MovieLensAnalysisReport.pdf 
###########################################################################

#This part of the code calculates mean of the train_set$rating column
#and stores the result in variable named "mu": Commented by Khaliun.B 2021.04.26

x<-train_set$pred
y<-train_set$rating

nrow(train_set)

x_test<-test_set$pred
y_test<-test_set$rating

summary(test_set)

fit_lm <-  lm(y ~ x)

y_hat_lm <- predict(fit_lm, as.factor(y_test))
levels(as.factor(y_hat_lm))
levels(y_test)
cm <- confusionMatrix(y_hat_lm, y_test)
cm$overall["Accuracy"]

#RMSE(fit, test_set$rating)
###########################################################################
### END: This chunk of code is training Linear model with Regularized Movie Effect + User Effect model 
###########################################################################