##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

###################################################################
### BEGIN: This group of code prepares edx and validation data sets
### Original code has been provided by edx course
###################################################################

# This part of the code installs required packages for the project if not installed previously: Commented by Khaliun.B 2021.04.11 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
#

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# This part of the code downloads the MovieLens 10M data: Commented by Khaliun.B 2021.04.11
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# This part of the code reads and populates "ratings" variable from downloaded MovieLens 10M data: Commented by Khaliun.B 2021.04.11
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

#  This part of the code reads and populates "movies" variable from downloaded MovieLens 10M data: Commented by Khaliun.B 2021.04.11
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# This part of the code creates movies data frame: Commented by Khaliun.B 2021.04.11.
# Using R 3.6. Commented by Khaliun.B 2021.04.11. Using R 3.6
# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

# This part of the code is commented out by Khaliun.B 2021.04.11. Using R 3.6
# if using R 4.0 or later:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
#                                           title = as.character(title),
#                                           genres = as.character(genres))

# This part of the code joins the movies and rating data sets and creates movielens data frame: Commented by Khaliun.B 2021.04.11
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
# This part of the code creates 90%:10% partitions for the MovieLens project and divides movielens data frame into edx and validation set: Commented by Khaliun.B 2021.04.11
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
#Final version of the validation set. Which is equivalent of test_set for this MovieLens project: Commented by Khaliun.B 2021.04.11
#Validation set hold 999999 rows and 6 columns Which is aproximately 11% of edx set 9000055 and 10% of the movielens data frame: Commented by Khaliun.B 2021.04.11
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)

#Final version of the edx set. Which is equivalent of training_set for this MovieLens project: Commented by Khaliun.B 2021.04.11
#edx set holds 9000055 rows and 6 colums. Which is 90% of the movielens data frame: Commented by Khaliun.B 2021.04.11
edx <- rbind(edx, removed)

# This part of the code removes variables used for "edx" and "validation" data sets  other than "edx" and "validation": Commented by Khaliun.B 2021.04.11
rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################################################
### END: This group of code prepares edx and validation data sets
### Original code has been provided by edx course
###################################################################

###################################################################
### BEGIN: This group of code prepares edx data set for training of lm()
### The comment has been added by Khaliun.B 2021.05.03
###################################################################

#This part of the code assigns lamdba value. lambda value has been previously tuned by analysis: Commented by Khaliun.B 2021.05.03
l <- 4.75

#This part of the code calculates bi and bu features to be used for training lm() model: Commented by Khaliun.B 2021.05.03 
mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

#This part of the code mutates bi, bu, pred features into the original edx data set for use of training lm() model: Commented by Khaliun.B 2021.05.03 

edx <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u)

#This part of the code mutates bi, bu, pred features into the original validation data set for use of evaluating lm() model results: Commented by Khaliun.B 2021.05.03 

validation <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u)

###################################################################
### END: This group of code prepares edx data set for training of lm()
### The comment has been added by Khaliun.B 2021.05.03
###################################################################

#This part of the code creates function for calculating Residual Mean Squared Error: Commented by Khaliun.B 2021.04.26
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

###################################################################
### BEGIN: This group of code prepares subsets of edx data set for analysis and testing of lm() execution results
### The comment has been added by Khaliun.B 2021.05.03
### (!) We don't need this group of code to run for the final RMSE
### Therefore all the code had been commented out with triple hashtag ###
###################################################################

#This part of the code divides movielens data into
# 80%:20% training set named "train_set" and test set named "train_set": Commented by Khaliun.B 2021.04.12
###set.seed(755)
#sample_edx<-sample_n(edx,100000)

###test_index <- createDataPartition(y = sample_edx$rating, times = 1,
###                                  p = 0.2, list = FALSE)
###train_set <- sample_edx[-test_index,]
###test_set <- sample_edx[test_index,]
#Code results for length(test_index); total length of test_index: [1] 20002
#Code results for dim(train_set); training set has 80'002 rows and 7 columns: [1] 80002     7
#Code results for dim(test_set); test set has 20'002 rows and 7 columns: [1] 20002     7

#This part of the code does the semi-joins test_set with training set first using movieId
#and second using userId: Commented by Khaliun.B 2021.04.12
###test_set <- test_set %>% 
###  semi_join(train_set, by = "movieId") %>%
###  semi_join(train_set, by = "userId")
#Code results for dim(test_set) after semi_joins with train_set: [1] 19331     7
#This process as excluded 671 rows that were present in training set from test set

###################################################################
### END: This group of code prepares subsets of edx data set for analysis and testing of lm() execution results
### The comment has been added by Khaliun.B 2021.05.03
###################################################################