#### RMSE calculation script for MovieLens project
#### This file holds the script that is based on the edx data set and will not use final validation data set
#### Created by Khaliun.B 2021.04.11
#### According to the MovieLens project instructions
#### (!) Second, you will train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set.
#### 25 points: RMSE < 0.86490

#### The final model follows the Movie + User Effects Model: Yu,i=μ+bi+bu+ϵu,i

library(dslabs)
library(tidyverse)
library(caret)
data("movielens")

#Original movielens data used for Machine Learning course demonstration has one extra column named "year". And also the columns are in different order than MovieLens 10K data used for this project: Commented by Khaliun.B 2021.04.12
head(movielens)

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

#This part of the code samples 100 random and unique userIds from movielens data used for Machine Learning course demonstration and puts the userIds in users variable of integer class: Commented by Khaliun.B 2021.04.12
users <- sample(unique(movielens$userId), 100)
#Code results are following: [1] 134 527 626 535 188 124

#Not sure what this part of the code does yet.
rafalib::mypar()

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

#This part of the code creates function for calculating Residual Mean Squared Error: Commented by Khaliun.B 2021.04.12
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
#Function is named RMSE():

#This part of the code calculates mu_hat value for the training set
#and stores displays the value: Commented by Khaliun.B 2021.04.12
mu_hat <- mean(train_set$rating)
mu_hat
#Code result is following: [1] 3.542793

#This part of the code calculates Residual Mean Squared Error for the test_set
#using function RMSE() with test_set$rating column and mu_hat variable as a parameter,
#stores the result in naive_rmse variable and displays the value: Commented by Khaliun.B 2021.04.12 
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse
#Code result is following: [1] 1.04822

#This part of the code replicated value 2.5 number of times equal to number of test_set rows
#In this case 19'331 times: Commented by Khaliun.B 2021.04.12
predictions <- rep(2.5, nrow(test_set))
#Code results: predictions variable is of numeric class and has 19'331 rows all of value 2.5

#This part of the code uses previously prepared predictions variable to calculate
#Residual Mean Squared Error for 2.5 rating: Commented by Khaliun.B 2021.04.12
RMSE(test_set$rating, predictions)
#Code result is following: [1] 1.489453

#This part of the code creates a tibble named "rmse_results": Commented by Khaliun.B 2021.04.12
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
#Code result is following: # A tibble: 1 x 2
#                           method            RMSE
#                           <chr>            <dbl>
#                             1 Just the average  1.05

#This part of the code predicts full movielens data set rating using userId linear model
#and stores results into variable named fit: Commented by Khaliun.B 2021.04.12
fit <- lm(rating ~ as.factor(userId), data = movielens)
#Code results worked for half minute on 2021.04.2 19:12 and returned 671 result values:
#Call:
#lm(formula = rating ~ as.factor(userId), data = movielens)
#Coefficients:
#  (Intercept)    as.factor(userId)2    as.factor(userId)3    as.factor(userId)4    as.factor(userId)5 
#2.550000             0.936842             1.018627             1.798039             1.360000 
#as.factor(userId)6   as.factor(userId)7   as.factor(userId)8   as.factor(userId)9  as.factor(userId)10 
#...

#This part of the code calculates mean of the train_set$rating column
#and stores the result in variable named "mu": Commented by Khaliun.B 2021.04.12
mu <- mean(train_set$rating)
#Code result is following: [1] 3.542793

#This part of the code groups training set by movieId and calculates mean of
#values of each row's rating subtracted by overall training set rating column mean
#and stores the results into column named b_i: Commented by Khaliun.B 2021.04.12
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
#Code results for head(movie_avgs): # A tibble: 6 x 2
#                                     movieId    b_i
#                                     <int>  <dbl>
#                                       1       1  0.259
#                                     ...
#Code results for str(movie_avgs): tibble [8,469 × 2] (S3: tbl_df/tbl/data.frame)
#                                   $ movieId: int [1:8469] 1 2 3 4 5 6 7 8 9 10 ...
#                                   $ b_i    : num [1:8469] 0.259 -0.158 -0.321 -1.243 -0.298 ...

#This part of the code creates qplot histogram from "movie_avgs" calculated previously: Commented by Khaliun.B 2021.04.12
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))
#Code result shows in a histogram in Plots tab  which will be included in final report
#Plot shows b_i value is of normal distribution

#This part of the code creates a data frame named "predicted_ratings"
#which sets value for each row in test_set as mean rating for current row's movieId.
#Basically this is a prediction based on movieId average: Commented by Khaliun.B 2021.04.12
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
#Code results for str(predicted_ratings): num [1:19331] 3.53 3.33 3.08 3.57 3.92 ...

#This part of the code calculates Residual Mean Squared Error
#for predicted_ratings compared against actual test_set$rating: Commented by Khaliun.B 2021.04.12
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
model_1_rmse
#Code result is following: [1] 0.9862839

#This part of the code adds a column to the previously created data frame
#named "rmse_results" which contained RMSE value for naive_rmse, a model which
#predicted ratings based on just the overall train_set$rating average: Commented by Khaliun.B 2021.04.12
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()
#Code result if following: |method             |      RMSE|
#                           |:------------------|---------:|
#                           |Just the average   | 1.0482202|
#                           |Movie Effect Model | 0.9862839|

#This part of the code shows a histogram of b_u values
#which are means of ratings grouped by userId who had given
#more than 100 ratings. We could call these users active rater users: Commented by Khaliun.B 2021.04.12
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")
#Code result shows in a histogram in Plots tab  which will be included in final report
#Plot shows rating average for active rater users is around 3.5
#and active users rate movies mostly between 3 and 4.5

#This part of the code calculates means of rating
# grouped by each userId in test_set
# also considering Movie Effect calculated previously
# and stores results into user_avgs data: Commented by Khaliun.B 2021.04.12
    # lm(rating ~ as.factor(movieId) + as.factor(userId))
user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
#Code result for head(user_avgs): # A tibble: 6 x 2
#                                 userId    b_u
#                                 <int>  <dbl>
#                                   1      1 -1.19 
#                                 ...

#This part of the code predicts ratings considering
# movieId deviaton average and userId average deviations: Commented by Khaliun.B 2021.04.12
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
#Code results for head(predicted_ratings): [1] 2.346876 2.145727 1.893474 2.379961 2.733962 3.695936

#This part of the code calculates Residual Mean Squared Error
#for predicted_ratings compared against actual test_set$rating
#considering Movie Effect and User Effect: Commented by Khaliun.B 2021.04.12
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
model_2_rmse
#Code result is following: [1] 0.8848688

#This part of the code adds a column to the previously created data frame
#named "rmse_results" which contained RMSE value for Movie Effect model for
#test_set from movielens data used for Machine Learning course demonstration: Commented by Khaliun.B 2021.04.12
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()
#Code result is following: |method                     |      RMSE|
#                         |:--------------------------|---------:|
#                         |Just the average           | 1.0482202|
#                         |Movie Effect Model         | 0.9862839|
#                         |Movie + User Effects Model | 0.8848688|

######## Wrapping up for 2021.04.12 Tested each code