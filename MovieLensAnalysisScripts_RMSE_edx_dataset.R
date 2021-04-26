#### RMSE calculation script for MovieLens project
#### This file holds the script that is based on the edx data set and will not use final validation data set
#### Created by Khaliun.B 2021.04.11
#### According to the MovieLens project instructions
#### (!) Second, you will train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set.
#### 25 points: RMSE < 0.86490

#### (!) You should split the edx data into separate training and test sets to design and test your algorithm.

#### (!) The validation data (the final hold-out test set) should NOT be used for training, developing, or selecting your algorithm and it should ONLY be used for evaluating the RMSE of your final algorithm.

#This part of the code added by Khaliun.B 2021.04.17. library factoextra is needed
#for fviz_cluster() function for creating cluster plots in
#file: MovieLensAnalysisScripts_RMSE_edx_dataset.R
#if(!require(factoextra)) install.packages("factoextra", repos = "http://cran.us.r-project.org")
#

source("MovieLensAnalysisScripts.R", local = knitr::knit_global())

#This part of the code shows distinct userId and distinct movieIds in movielens data: Commented by Khaliun.B 2021.04.26
edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
#Code results are following:  n_users n_movies
#                             1   69878    10677
# (Note: results of the code vary with the partitioning process of the MovieLens 10M data. This result may not match with the pdf report and the current code run)

#This part of the code lists top 5 most rated movies in movielens data: Commented by Khaliun.B 2021.04.26
keep <- edx %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)
#Code results for keep: [1] 110 150 260 296 457
# (Note: results of the code vary with the partitioning process of the MovieLens 10M data. This result may not match with the pdf report and the current code run)

#This part of the code lists top 5 most rater movies in edx data: Commented by Khaliun.B 2021.04.26
keep_raters<-edx %>% filter(movieId %in% keep) %>%
    mutate(rating=1) %>%
    select(movieId,userId,rating) %>%
    group_by(userId) %>%
    summarize(n=sum(rating)) %>%
    head()%>%
    pull(userId)
  
#This part of the code filters userIds between 13:20 and top 5 most rated movies in edx data used for Machine Learning course demonstration
# and transposes the title and rating columns by value and lists the results for userId column: Commented by Khaliun.B 2021.04.12
tab <- edx %>%
  filter(userId %in% keep_raters) %>% 
  filter(movieId %in% keep) %>%
  arrange(userId) %>%
  select(userId, title, rating) %>% 
  spread(title, rating)

#This part of the converts the results of the "tab" data frame and turnes it into knitr_kable format, which is character string by default. Data is processed based on edx data used for Machine Learning course demonstration: Commented by Khaliun.B 2021.04.12
tab %>% knitr::kable()
# (Note: results of the code vary with the partitioning process of the MovieLens 10M data. This result may not match with the pdf report and the current code run)

#This part of the code samples 100 random and unique userIds from edx data: Commented by Khaliun.B 2021.04.26
users <- sample(unique(edx$userId), 100)
#Code results for head(users): [1] 34961 22764 25566 24809 58003 61596
# (Note: results of the code vary with the partitioning process of the MovieLens 10M data. This result may not match with the pdf report and the current code run)

#This part of the code creates an image showing the spread across 100 movies and users rated at least once.
#Values for the image are filtered using "users" variable integer values (which are randomly sampled 100 values)
# from edx data: Commented by Khaliun.B 2021.04.26
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users") 

abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
#Code result shows in an image in a Plots tab which will be included in final report

#This part of the code counts number of times each movieId is rated and creates geom_histogram
#which scales the number of movies by log10: Commented by Khaliun.B 2021.04.26
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
#Code result shows in a geom_histogram named "Movies" and with x axis named "n" which are number of movieIds rated
#against y axis named "count" which are times those movieIds had been rated
#in a Plots tab which will be included in final report.

#This part of the code counts number of times each userId is rated and creates geom_histogram
#which scales the number of users who had given ratings by log10: Commented by Khaliun.B 2021.04.26
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")
#Code result shows in a geom_histogram named "Users" and with x axis named "n" which are
#number of userIds of users who had given ratings for a movie
#against y axis named "count" which are number of ratings those userIds had given for movies
#in a Plots tab which will be included in final report.
#Results of the plot shows that values of y axis drops sharply for x axis value of 300.
#Below x axis value of 300, y axis value drops are gradual.
#This means most users rarely give above 300 ratings in total.
#And users who rate the movies usually give below 100 ratings.

#This part of the code creates sequence of lamdba values for parameter tuning: Commented by Khaliun.B 2021.04.13
lambdas <- seq(0, 10, 0.25)
#Code result for head(lambdas): [1] 0.00 0.25 0.50 0.75 1.00 1.25

#This part of the code creates function for calculating Residual Mean Squared Error: Commented by Khaliun.B 2021.04.12
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#This part of the code calculates mean of the train_set$rating column
#and stores the result in variable named "mu": Commented by Khaliun.B 2021.04.26
mu <- mean(train_set$rating)

#This part of the code calculates RMSE on the test set for each
#value of "lambdas" sequence as Regularized Movie + User Effect
#on the training set and stores results in "rmses" variable: Commented by Khaliun.B 2021.04.26
rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

#This part of the code creates qplot of lambdas against rmses: Commented by Khaliun.B 2021.04.26
qplot(lambdas, rmses)
#Code result shows in a qplot in Plots tab  which will be included in final report

#This part of the code shows value of lambda correspoding to minimum value of rmses
#and stores the minimum value in variable "lambda": Commented by Khaliun.B 2021.04.26
lambda <- lambdas[which.min(rmses)]
lambda
#Code result is following: [1] 4.75

#This part of the code creates data frame "rmse_results" which contains RMSE value
# Regularized Movie+User Effect Model RMSE: Commented by Khaliun.B 2021.04.26
rmse_results <-data_frame(method="Regularized Movie + User Effect Model",
                                     RMSE = min(rmses))
rmse_results %>% knitr::kable()
#########################################

########################################
