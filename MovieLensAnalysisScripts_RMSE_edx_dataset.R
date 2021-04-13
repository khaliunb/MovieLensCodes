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
#We don't need to train datasets just yet. Therefore commenting the code: Commented by Khaliun.B 2021.04.13
#fit <- lm(rating ~ as.factor(userId), data = movielens)
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

#### Wrapping up for 2021.04.12 Tested each code

#### Commented by Khaliun.B 2021.04.13
#### This part of the code was copied from Machine Learning course
#### Course/Section 6: Model Fitting and Recommendation Systems/6.3: Regularization


#This part of the code displays movie title for 10 movies that have largest 
# absolute value of deviations from predictions with Movie Effect model
# in the test_set. Which are named residual: Commented by Khaliun.B 2021.04.13
test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% knitr::kable()
#Code result is following:
# |title                                            |  residual|
#  |:------------------------------------------------|---------:|
#  |Day of the Beast, The (Día de la Bestia, El)     |  4.500000|
#  |Horror Express                                   | -4.000000|
#  |No Holds Barred                                  |  4.000000|
#  |Dear Zachary: A Letter to a Son About His Father | -4.000000|
#  |Faust                                            | -4.000000|
#  |Hear My Song                                     | -4.000000|
#  |Confessions of a Shopaholic                      | -4.000000|
#  |Twilight Saga: Breaking Dawn - Part 1, The       | -4.000000|
#  |Taxi Driver                                      | -3.806931|
#  |Taxi Driver                                      | -3.806931|
# As we can see, movies that have largest residual error are mostly obscure

#This part of the code selects distinct movieId and title from full movielens dataset
#and saves them into movie_titles variable: Commented by Khaliun.B 2021.04.13
movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()
str(movie_titles)
#Code results for str(movie_titles): 'data.frame':	9066 obs. of  2 variables:
#                                     $ movieId: int  31 1029 1061 1129 1172 1263 1287 1293 1339 1343 ...
#                                     $ title  : chr  "Dangerous Minds" "Dumbo" "Sleepers" "Escape from New York" ... 

#This part of the code joins movie_avgs and movie_titles, lists the results in descending order
# by b_i and displays first 10 movie titles and b_i. The result is movie titles with largest b_i: Commented by Khaliun.B 2021.04.13
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()
#Code result is following:
#|title                                                   |      b_i|
#|:-------------------------------------------------------|--------:|
#|Lamerica                                                | 1.457207|
#|Love & Human Remains                                    | 1.457207|
#|Enfer, L'                                               | 1.457207|
#|Picture Bride (Bijo photo)                              | 1.457207|
#|Red Firecracker, Green Firecracker (Pao Da Shuang Deng) | 1.457207|
#|Faces                                                   | 1.457207|
#|Maya Lin: A Strong Clear Vision                         | 1.457207|
#|Heavy                                                   | 1.457207|
#|Gate of Heavenly Peace, The                             | 1.457207|
#|Death in the Garden (Mort en ce jardin, La)             | 1.457207|
# As we can see from the titles, movies that have largest b_i value are also obscure

#This part of the code joins movie_avgs and movie_titles, lists the results in descending order
# by b_i and displays first 10 movie titles and b_i. The result is movie titles with lowest b_i: Commented by Khaliun.B 2021.04.13
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()
#Code result is following:
#|title                                        |       b_i|
#|:--------------------------------------------|---------:|
#|Santa with Muscles                           | -3.042793|
#|B*A*P*S                                      | -3.042793|
#|3 Ninjas: High Noon On Mega Mountain         | -3.042793|
#|Barney's Great Adventure                     | -3.042793|
#|Merry War, A                                 | -3.042793|
#|Day of the Beast, The (Día de la Bestia, El) | -3.042793|
#|Children of the Corn III                     | -3.042793|
#|Whiteboyz                                    | -3.042793|
#|Catfish in Black Bean Sauce                  | -3.042793|
#|Watcher, The                                 | -3.042793|
# As we can see from the titles, movies that have lowest b_i value are also obscure

#This part of the code joins movie_avgs and movie_titles, lists the results in descending order
# by b_i and displays first 10 movie titles, b_i and
#number of times that movie had been rated: Commented by Khaliun.B 2021.04.13
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()
#Code result is following:
#|title                                                   |      b_i|  n|
#|:-------------------------------------------------------|--------:|--:|
#|Lamerica                                                | 1.457207|  1|
#|Love & Human Remains                                    | 1.457207|  3|
#|Enfer, L'                                               | 1.457207|  1|
#|Picture Bride (Bijo photo)                              | 1.457207|  1|
#|Red Firecracker, Green Firecracker (Pao Da Shuang Deng) | 1.457207|  3|
#|Faces                                                   | 1.457207|  1|
#|Maya Lin: A Strong Clear Vision                         | 1.457207|  2|
#|Heavy                                                   | 1.457207|  1|
#|Gate of Heavenly Peace, The                             | 1.457207|  1|
#|Death in the Garden (Mort en ce jardin, La)             | 1.457207|  1|
#As we can see, these are all movies that have been rated very few times

#This part of the code joins movie_avgs and movie_titles, lists the results in ascending order
# by b_i and displays first 10 movie titles, b_i and
#number of times that movie had been rated: Commented by Khaliun.B 2021.04.13
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()
#Code result is following:
#|title                                        |       b_i|  n|
#|:--------------------------------------------|---------:|--:|
#|Santa with Muscles                           | -3.042793|  1|
#|B*A*P*S                                      | -3.042793|  1|
#|3 Ninjas: High Noon On Mega Mountain         | -3.042793|  1|
#|Barney's Great Adventure                     | -3.042793|  1|
#|Merry War, A                                 | -3.042793|  1|
#|Day of the Beast, The (Día de la Bestia, El) | -3.042793|  1|
#|Children of the Corn III                     | -3.042793|  1|
#|Whiteboyz                                    | -3.042793|  1|
#|Catfish in Black Bean Sauce                  | -3.042793|  1|
#|Watcher, The                                 | -3.042793|  1|
# As we can see from the titles, movies that have lowest b_i value all have been rated just one time 

#This part of the code sets value 3 to variable named "lambda": Commented by Khaliun.B 2021.04.13
lambda <- 3

#This part of the code calculates mean of rating column from train_set and assigns the
# value to variable mu: Commented by Khaliun.B 2021.04.13
mu <- mean(train_set$rating)
#Code result is following: [1] 3.542793

#This part of the code groups train_set by movieId and calculates b_i considering
#total number of times movie had been rated and lambda. Result is assigned to
#data frame called movie_reg_avgs: Commented by Khaliun.B 2021.04.13
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
#Code results for head(movie_reg_avgs): # A tibble: 6 x 3
#                                       movieId    b_i   n_i
#                                       <int>  <dbl> <int>
#                                         1       1  0.255   189
#Code results for class(movie_reg_avgs): [1] "tbl_df"     "tbl"        "data.frame"

#This part of the code creates plot that shows distribution of regularized b_i against original b_i
#circle sizes represent the times movies have been rated: Commented by Khaliun.B 2021.04.13
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)
#Code result shows in a geom_points in Plots tab  which will be included in final report

#This part of the code joins movie_reg_avgs (which in this case is a regularized model with lambda value of 3)
# and movie_titles, lists the results in descending order
# by b_i and displays first 10 movie titles, b_i and
#number of times that movie had been rated: Commented by Khaliun.B 2021.04.13
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()
#Code result is following:
#|title                          |       b_i|   n|
#|:------------------------------|---------:|---:|
#|All About Eve                  | 0.9271514|  26|
#|Shawshank Redemption, The      | 0.9206986| 240|
#|Godfather, The                 | 0.8971328| 153|
#|Godfather: Part II, The        | 0.8710751| 100|
#|Maltese Falcon, The            | 0.8597749|  47|
#|Best Years of Our Lives, The   | 0.8592343|  11|
#|On the Waterfront              | 0.8467603|  23|
#|Face in the Crowd, A           | 0.8326899|   4|
#|African Queen, The             | 0.8322939|  36|
#|All Quiet on the Western Front | 0.8235200|  11|

#This part of the code joins movie_reg_avgs (which in this case is a penalised model with lambda value of 3)
# and movie_titles, lists the results in ascending order
# by b_i and displays first 10 movie titles, b_i and
#number of times that movie had been rated: Commented by Khaliun.B 2021.04.13
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()
#Code result is following:
#|title                              |       b_i|  n|
#|:----------------------------------|---------:|--:|
#|Battlefield Earth                  | -2.064653| 14|
#|Joe's Apartment                    | -1.779955|  7|
#|Speed 2: Cruise Control            | -1.689385| 20|
#|Super Mario Bros.                  | -1.597269| 13|
#|Police Academy 6: City Under Siege | -1.571379| 10|
#|After Earth                        | -1.524453|  4|
#|Disaster Movie                     | -1.521396|  3|
#|Little Nicky                       | -1.511374| 17|
#|Cats & Dogs                        | -1.472973|  6|
#|Blade: Trinity                     | -1.462194| 11|
#As we can see, penalized model's errors are now much better
#The total number of ratings per movie are higher than Movie Effect+User Effect model
#Also the movies are relatively well known

#This part of the code predicts ratings with b_i value considering
# movieId deviaton average and userId average deviations
# and penalized by lambda value 3. Regularized version: Commented by Khaliun.B 2021.04.13
predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred
#Code results for head(predicted_ratings): [1] 3.535262 3.349446 3.115709 3.565709 3.900526 3.870110

#This part of the code calculates Residual Mean Squared Error
#for regularized version of predicted_ratings compared against actual test_set$rating
#considering Movie Effect: Commented by Khaliun.B 2021.04.13
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
model_3_rmse
#Code result if following: [1] 0.9649457

#This part of the code adds a column to the previously created data frame
#named "rmse_results" which contained RMSE value for Movie Effect model and User Effect model for
#test_set from movielens data used for Machine Learning course demonstration
#and adds Regularized Movie Effect Model RMSE: Commented by Khaliun.B 2021.04.13
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()
#Code result is following:
#|method                         |      RMSE|
#|:------------------------------|---------:|
#|Just the average               | 1.0482202|
#|Movie Effect Model             | 0.9862839|
#|Movie + User Effects Model     | 0.8848688|
#|Regularized Movie Effect Model | 0.9649457|
#As we can see, Regularized Movie Effect Model RMSE went up because it just considers Movie Effect model
#But lower than non-regularized Movie Effect Model

#This part of the code creates sequence of lamdba values for parameter tuning: Commented by Khaliun.B 2021.04.13
lambdas <- seq(0, 10, 0.25)
#Code results for head(lambdas): [1] 0.00 0.25 0.50 0.75 1.00 1.25

#This part of the code calculates mean of rating column from train_set and assigns the
# value to variable mu: Commented by Khaliun.B 2021.04.13
mu <- mean(train_set$rating)
mu
#Code result is following: [1] 3.542793

#This part of the code calculates sum of the rating-mu, total number of ratings for each movieId
#and stores the result in data frame named "just_the_sum": Commented by Khaliun.B 2021.04.13
just_the_sum <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
head(just_the_sum)
#Code result for head(just_the_sum):  # A tibble: 6 x 3
#                                     movieId     s   n_i
#                                     <int> <dbl> <int>
#                                       1       1  48.9   189
#                                     2       2 -12.3    78
#                                     3       3 -14.4    45
#                                     4       4 -12.4    10
#                                     5       5 -14.0    47
#                                     6       6  32.0    82
#                                     ...

#This part of the code calculates RMSE on the test set for each
#value of "lambdas" sequence as Regularized Movie Effect Model
#and stores results in "rmses" variable: Commented by Khaliun.B 2021.04.13
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
head(rmses)
#Code result for head(rmses): [1] 0.9862839 0.9782502 0.9736557 0.9707808 0.9688783 0.9675754

#This part of the code creates qplot of lambdas against rmses: Commented by Khaliun.B 2021.04.13
qplot(lambdas, rmses)  
#Code result shows in a qplot in Plots tab  which will be included in final report

#This part of the code shows value of lambda correspoding to minimum value of rmses: Commented by Khaliun.B 2021.04.13
lambdas[which.min(rmses)]
#Code result is following: [1] 3

#This part of the code creates sequence of lamdba values for parameter tuning: Commented by Khaliun.B 2021.04.13
lambdas <- seq(0, 10, 0.25)
#Code result for head(lambdas): [1] 0.00 0.25 0.50 0.75 1.00 1.25

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
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
head(rmses)
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
#Code result is following: |method                                |      RMSE|
#                           |:-------------------------------------|---------:|
#                           |Just the average                      | 1.0482202|
#                           |Movie Effect Model                    | 0.9862839|
#                           |Movie + User Effects Model            | 0.8848688|
#                           |Regularized Movie Effect Model        | 0.9649457|
#                           |Regularized Movie + User Effect Model | 0.8806419|

#### Wrapping up for 2021.04.13 Tested each code
#### Final RMSE is 0.8806419
#### We need RMSE < 0.86490 for full RMSE points
#### Maybe I need to add the half points effect