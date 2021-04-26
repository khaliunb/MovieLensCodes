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

library(dslabs)

#This part of the code picks 10'000 random samples from edx data set and assigns them to movielens data set
movielens <- sample(edx,10000,replace=TRUE)


#This part of the code shows distinct userId and distinct movieIds in movielens data used for Machine Learning course demonstration: Commented by Khaliun.B 2021.04.12
movielens %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
#Code results are following:  n_users n_movies
#                             1     671     9066

#This part of the code lists top 5 most rated movies in movielens data used for Machine Learning course demonstration: Commented by Khaliun.B 2021.04.12
keep <- movielens %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)
head(keep)
#Code results are following: [1] 260 296 318 356 593

#This part of the code lists filters userIds between 13:20 and top 5 most rated movies in movielens data used for Machine Learning course demonstration
# and transposes the title and rating columns by value and lists the results for userId column: Commented by Khaliun.B 2021.04.12
tab <- movielens %>%
  filter(userId %in% c(13:20)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)
#Code results are following:  userId Forrest Gump Pulp Fiction Shawshank Redemption, The Silence of the Lambs, The Star Wars: Episode IV - A New Hope
#                             1     13          5.0          3.5                       4.5                        NA                                 NA
#                             2     15          1.0          5.0                       2.0                       5.0                                5.0
#                             ...

#This part of the converts the results of the "tab" data frame and turnes it into knitr_kable format, which is character string by default. Data is processed based on movielens data used for Machine Learning course demonstration: Commented by Khaliun.B 2021.04.12
tab %>% knitr::kable()
#Code results are following:| userId| Forrest Gump| Pulp Fiction| Shawshank Redemption, The| Silence of the Lambs, The| Star Wars: Episode IV - A New Hope|
#                           |------:|------------:|------------:|-------------------------:|-------------------------:|----------------------------------:|
#                           |     13|          5.0|          3.5|                       4.5|                        NA|                                 NA|
#                           ...

rafalib::mypar()

#This part of the code samples 100 random and unique userIds from movielens data used for Machine Learning course demonstration and puts the userIds in users variable of integer class: Commented by Khaliun.B 2021.04.12
users <- sample(unique(movielens$userId), 100)
#Code results are following: [1] 134 527 626 535 188 124

#This part of the code creates an image showing the spread across 100 movies and users rated at least once.
#Values for the image are filtered using "users" variable integer values (which are randomly sampled 100 values)
# from movielens data used for Machine Learning course demonstration: Commented by Khaliun.B 2021.04.12
movielens %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
#Code result shows in an image in a Plots tab which will be included in final report

#This part of the code counts number of times each movieId is rated and creates geom_histogram
#which scales the number of movies by log10: Commented by Khaliun.B 2021.04.12
movielens %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
#Code result shows in a geom_histogram named "Movies" and with x axis named "n" which are number of movieIds rated
#against y axis named "count" which are times those movieIds had been rated
#in a Plots tab which will be included in final report.
#Results of the plot shows that with the increase of movieId numbers rated, drops the number of ratings per movieId
#Which means most movies are rated only 1 time.

#This part of the code counts number of times each userId is rated and creates geom_histogram
#which scales the number of users who had given ratings by log10: Commented by Khaliun.B 2021.04.12
movielens %>%
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

#This part of the code divides movielens data used for Machine Learning course demonstration into
# 80%:20% training set named "train_set" and test set named "train_set": Commented by Khaliun.B 2021.04.12
set.seed(755)
test_index <- createDataPartition(y = movielens$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]
#Code results for length(test_index); total length of test_index: [1] 20002
#Code results for dim(train_set); training set has 80'002 rows and 7 columns: [1] 80002     7
#Code results for dim(test_set); test set has 20'002 rows and 7 columns: [1] 20002     7

#This part of the code does the semi-joins test_set with training set first using movieId
#and second using userId: Commented by Khaliun.B 2021.04.12
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
#Code results for dim(test_set) after semi_joins with train_set: [1] 19331     7
#This process as excluded 671 rows that were present in training set from test set

#This part of the code creates sequence of lamdba values for parameter tuning: Commented by Khaliun.B 2021.04.13
lambdas <- seq(0, 10, 0.25)
#Code result for head(lambdas): [1] 0.00 0.25 0.50 0.75 1.00 1.25
summary(train_set)

#This part of the code calculates RMSE on the test set for each
#value of "lambdas" sequence as Regularized Movie + User Effect
#on the training set and stores results in "rmses" variable: Commented by Khaliun.B 2021.04.13
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  fit <- lm(rating ~ as.factor(userId), data = movielens)
    predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
#Code result for head(rmses): [1] 0.9077043 0.8981453 0.8925766 0.8890183 0.8866047 0.8849018

#This part of the code creates qplot of lambdas against rmses: Commented by Khaliun.B 2021.04.13
qplot(lambdas, rmses)  
#Code result shows in a qplot in Plots tab  which will be included in final report

#This part of the code shows value of lambda correspoding to minimum value of rmses
#and stores the minimum value in variable "lambda": Commented by Khaliun.B 2021.04.13
lambda <- lambdas[which.min(rmses)]
lambda
#Code result is following: [1] 3.75

#This part of the code adds a column to the previously created data frame
#named "rmse_results" which contained RMSE value for Movie Effect model, User Effect model,
#Regularized Movie effect model for test_set from movielens data used for Machine Learning
#course demonstration and adds Regularized Movie+User Effect Model RMSE: Commented by Khaliun.B 2021.04.13
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

#########################################