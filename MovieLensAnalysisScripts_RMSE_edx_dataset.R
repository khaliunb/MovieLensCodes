#### RMSE calculation script for MovieLens Capstone project
#### This file holds the script that is based on the edx data set and will not use final validation data set
#### Created by Khaliun.B 2021.04.11

########### CHECKLIST FROM INSTRUCTIONS to follow: ########################
#### According to the MovieLens project instructions
#### (!) Second, you will train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set.
#### 25 points: RMSE < 0.86490

#### (!) You should split the edx data into separate training and test sets to design and test your algorithm: Done
#### (!) The validation data (the final hold-out test set) should NOT be used for training, developing, or selecting your algorithm and it should ONLY be used for evaluating the RMSE of your final algorithm: Done
###########################################################################

###########################################################################
### BEGIN: This chunk of code is preparing data for the project
### Put the data preparation script into separate file named MovieLensAnalysisScripts.R, so that I can reuse the data preparation script in the RMSE, analysis and Final report RMD files. 
### Summary of the data used for the project are also included in Final Report. Please refer to file MovieLensAnalysisReport.pdf 
###########################################################################

#This part of the code records a start time for script running timing log
#Code has been commented out with triple hashtag ###: : Commented by Khaliun.B 2021.05.04
###dpST_TM<-Sys.time()
#

#This part of the code runs data preparation script from R script file MovieLensAnalysisScripts.R
# Note: Downloading and running script takes a few minutes
source("MovieLensAnalysisScripts.R", local = knitr::knit_global())

#This part of the code records a end time for script running timing log
#Code has been commented out with triple hashtag ###: : Commented by Khaliun.B 2021.05.04
###dpED_TM<-Sys.time()
#

#This part of the code creates a timing log for Data source prep script run time
#Code has been commented out with triple hashtag ###: : Commented by Khaliun.B 2021.05.04
###timing_results <-data_frame(log="Data source prep script run time",
###                            ST_TM = dpST_TM,
###                            ED_TM = dpED_TM,
###                            DURATION=difftime(dpED_TM,dpST_TM,units = "secs"))
###timing_results %>% select(log,DURATION) %>% knitr::kable()
#Results will be used for Final Report

###########################################################################
### END: This chunk of code is preparing data for the project
###########################################################################

###########################################################################
### BEGIN: This chunk of code is training Linear model with Regularized Movie Effect + User Effect model
### on edx data set and compares the prediction results with validation data set to calculate final RMSE.
### (!) We are using lamdbda value of 4.75 gained from analysis of edx data
### For complete tuning process of lambda value please refer to code file MovieLensAnalysisScript_DataExploration.R
### Analysis results are also included in Final Report. Please refer to file MovieLensAnalysisReport.pdf 
###########################################################################

#This part of the code records a start time for script running timing log
#Code has been commented out with triple hashtag ###: : Commented by Khaliun.B 2021.05.04
###trST_TM<-Sys.time()
#

#This part of the code trains edx data set with lm() model and stores results into variable fit_lm: Commented by Khaliun.B 2021.05.04
fit_lm <-  lm(rating ~ pred,data=edx)
#This part of the code predicts the rating using validation data set and fit_lm model and stores results into variable predicted_lm: Commented by Khaliun.B 2021.05.04
predicted_lm <- predict(fit_lm, newdata=validation)

#This part of the code records a end time for script running timing log
#Code has been commented out with triple hashtag ###: : Commented by Khaliun.B 2021.05.04
###trED_TM<-Sys.time()
#

#This part of the code creates a timing log for Data source prep script run time
#Code has been commented out with triple hashtag ###: : Commented by Khaliun.B 2021.05.04
###timing_results <-bind_rows(timing_results,data_frame(log="Edx Data set training script run time",
###                                                     ST_TM = trST_TM,
###                                                     ED_TM = trED_TM,
###                                                     DURATION=difftime(trED_TM,trST_TM,units = "secs")))
###timing_results %>% select(log,DURATION) %>% knitr::kable()
#Results will be used for Final Report

#This part of the code calculates RMSE for predictions and validation$rating
RMSE(validation$rating, predicted_lm)
#Final RMSE value: [1] 0.8648617

###########################################################################
### END: This chunk of code is training Linear model with Regularized Movie Effect + User Effect model 
###########################################################################