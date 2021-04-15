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
#### Maybe I need to add the half points effect?

#### Commented by Khaliun.B 2021.04.14
#### This part of the code was copied from Machine Learning course
#### Course/Section 6: Model Fitting and Recommendation Systems/6.3: Regularization
#### Matrix Factorization: Adresses example of pairing of values

#This part of the code filters users users who have rated movies above 50 times (active voters)
#who had rated specific movieId #3252 (Scent of a Woman used in example)
#saves the results into data frame called "train_small": Commented by Khaliun.B 2021.04.14
train_small <- movielens %>% 
  group_by(movieId) %>%
  filter(n() >= 50 | movieId == 3252) %>% ungroup() %>%
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup()
#Code results for head(train_small) are following: # A tibble: 6 x 7
#                                                   movieId title                  year genres                    userId rating timestamp
#                                                   <int> <chr>                 <int> <fct>                      <int>  <dbl>     <int>
#                                                     1      10 GoldenEye              1995 Action|Adventure|Thriller      2      4 835355493
#                                                     2      17 Sense and Sensibility  1995 Drama|Romance                  2      5 835355681
#                                                     3      39 Clueless               1995 Comedy|Romance                 2      5 835355604
#                                                   ...

#This part of the code selects userId, movieId and rating columns
#transposes movieId column and sets rating as value for the cells,
#converts result into matrix and stores the data in "y": Commented by Khaliun.B 2021.04.14
y <- train_small %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()
#Code results for dim(y): [1] 292 455

#This part of the code names rows of matrix "y" by the first column values: Commented by Khaliun.B 2021.04.14
rownames(y)<- y[,1]

#This part of the code reassigns columns and rows of matrix "y" to itself
#except for the first column. Excludes duplicates of row names
y <- y[,-1]

#This part of the code names columns of matrix "y" with the matching movie title
colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])

#This part of the code subtracts row means of matrix y from matrix y
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))

#This part of the code subtracts column means of matrix y from matrix y
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))

#This part of the code plots values for movies "Godfather, The" on axis x against
# and "Godfather: Part II, The" on axis y
m_1 <- "Godfather, The"
m_2 <- "Godfather: Part II, The"
qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)
#Resulting plot shows definite correlation between the two movies.

#This part of the code plots values for movies "Godfather, The" on axis x against
# and "Goodfellas" on axis y
m_1 <- "Godfather, The"
m_3 <- "Goodfellas"
qplot(y[ ,m_1], y[,m_3], xlab = m_1, ylab = m_3)
#Resulting plot shows definite correlation between the two movies.

#This part of the code plots values for movies "You've Got Mail" on axis x against
# and "Sleepless in Seattle" on axis y
m_4 <- "You've Got Mail" 
m_5 <- "Sleepless in Seattle" 
qplot(y[ ,m_4], y[,m_5], xlab = m_4, ylab = m_5)
#Resulting plot shows definite correlation between Meg Ryan movies.

#This part of the code shows correlation between 
#"Godfather, The","Godfather: Part II, The","Goodfellas","You've Got Mail","Sleepless in Seattle"
#movie ratings 
cor(y[, c(m_1, m_2, m_3, m_4, m_5)], use="pairwise.complete") %>% 
  knitr::kable()
#Code results are following:
#|                        | Godfather, The| Godfather: Part II, The| Goodfellas| You've Got Mail| Sleepless in Seattle|
#|:-----------------------|--------------:|-----------------------:|----------:|---------------:|--------------------:|
#|Godfather, The          |      1.0000000|               0.8320756|  0.4541425|      -0.4535093|           -0.3540335|
#|Godfather: Part II, The |      0.8320756|               1.0000000|  0.5400558|      -0.3377691|           -0.3259897|
#|Goodfellas              |      0.4541425|               0.5400558|  1.0000000|      -0.4894054|           -0.3672836|
#|You've Got Mail         |     -0.4535093|              -0.3377691| -0.4894054|       1.0000000|            0.5423584|
#|Sleepless in Seattle    |     -0.3540335|              -0.3259897| -0.3672836|       0.5423584|            1.0000000|
#As we can see, there is reverse correlation between group of movies
#Group1: "Godfather, The","Godfather: Part II, The","Goodfellas" - Al Pacino movies
#Group2: "You've Got Mail","Sleepless in Seattle" - Meg Ryan movies

set.seed(1)
options(digits = 2)

#This part of the code creates matrix Q with 1 column
# and names the rows by the names of the movies "Godfather, The","Godfather: Part II, The",
#"Goodfellas","You've Got Mail","Sleepless in Seattle"
Q <- matrix(c(1 , 1, 1, -1, -1), ncol=1)
rownames(Q) <- c(m_1, m_2, m_3, m_4, m_5)

#This part of the code creates matrix P with 1 column,
#fills the rows replicating: value 2 by 3 times, value 0 by 5 times and value -2 by 4 times 
#and names the rows with their ordered number
P <- matrix(rep(c(2,0,-2), c(3,5,4)), ncol=1)
rownames(P) <- 1:nrow(P)

#This part of the code multiplies transposed version of matrix Q
#with matrix P and stores resulting matrix as "X"
X <- jitter(P%*%t(Q))
X %>% knitr::kable(align = "c")
#Code results are following:| Godfather, The | Godfather: Part II, The | Goodfellas | You've Got Mail | Sleepless in Seattle |
#                           |:--------------:|:-----------------------:|:----------:|:---------------:|:--------------------:|
#                           |      1.81      |          2.15           |    1.81    |      -1.76      |        -1.81         |
#                           |      1.90      |          1.91           |    1.91    |      -2.31      |        -1.85         |
#                           |      2.06      |          2.22           |    1.61    |      -1.82      |        -2.02         |
#                           |      0.33      |          0.00           |   -0.09    |      -0.07      |         0.29         |
#                           |     -0.24      |          0.17           |    0.30    |      0.26       |        -0.05         |
#                           |      0.32      |          0.39           |   -0.13    |      0.12       |        -0.20         |
#                           |      0.36      |          -0.10          |   -0.01    |      0.23       |        -0.34         |
#                           |      0.13      |          0.22           |    0.08    |      0.04       |        -0.32         |
#                           |     -1.90      |          -1.65          |   -2.01    |      2.02       |         1.85         |
#                           |     -2.35      |          -2.23          |   -2.25    |      2.23       |         2.01         |
#                           |     -2.24      |          -1.88          |   -1.74    |      1.62       |         2.13         |
#                           |     -2.26      |          -2.30          |   -1.87    |      1.98       |         1.93         |
#Code results for dim(X): [1] 12  5

#This part of the code shows correlation of matrix X
cor(X) %>% knitr::kable(align="c")
#Code result is following: |                        | Godfather, The | Godfather: Part II, The | Goodfellas | You've Got Mail | Sleepless in Seattle |
#                          |:-----------------------|:--------------:|:-----------------------:|:----------:|:---------------:|:--------------------:|
#                          |Godfather, The          |      1.00      |          0.99           |    0.98    |      -0.98      |        -0.99         |
#                          |Godfather: Part II, The |      0.99      |          1.00           |    0.99    |      -0.98      |        -0.99         |
#                          |Goodfellas              |      0.98      |          0.99           |    1.00    |      -0.99      |        -0.99         |
#                          |You've Got Mail         |     -0.98      |          -0.98          |   -0.99    |      1.00       |         0.98         |
#                          |Sleepless in Seattle    |     -0.99      |          -0.99          |   -0.99    |      0.98       |         1.00         |

#This part of the code shows transposed matrix Q
t(Q) %>% knitr::kable(align="c")
#Code result is following: | Godfather, The | Godfather: Part II, The | Goodfellas | You've Got Mail | Sleepless in Seattle |
#                          |:--------------:|:-----------------------:|:----------:|:---------------:|:--------------------:|
#                          |       1        |            1            |     1      |       -1        |          -1          |

#This part of the code shows matrix P, transposed for viewing purpose
t(P) %>% knitr::kable(align="r")
#Code result is following: |  1|  2|  3|  4|  5|  6|  7|  8|  9| 10| 11| 12|
#                          |--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
#                          |  2|  2|  2|  0|  0|  0|  0|  0| -2| -2| -2| -2|

set.seed(1)
options(digits = 2)

#This part of the code repopulates matrix Q with 1 column
# and names the rows by the names of the movies "Godfather, The","Godfather: Part II, The",
#"Goodfellas","You've Got Mail","Sleepless in Seattle","Scent of a Woman"
m_6 <- "Scent of a Woman"
Q <- cbind(c(1 , 1, 1, -1, -1, -1), 
           c(1 , 1, -1, -1, -1, 1))
rownames(Q) <- c(m_1, m_2, m_3, m_4, m_5, m_6)

#This part of the code creates matrix P with 1 column,
#fills the rows replicating: value 2 by 3 times, value 0 by 5 times and value -2 by 4 times
#adds an extra column with values c(-1,1,1,0,0,1,1,1,0,-1,-1,-1),
#divides all the values by 2
#and names the rows with ordered number of rows from matrix X
P <- cbind(rep(c(2,0,-2), c(3,5,4)), 
           c(-1,1,1,0,0,1,1,1,0,-1,-1,-1))/2
rownames(P) <- 1:nrow(X)

#This part of the code multiplies transposed version of matrix Q
#with matrix P and stores resulting matrix as "X"
X <- jitter(P%*%t(Q), factor=1)
X %>% knitr::kable(align = "c")
#Code results are following:| Godfather, The | Godfather: Part II, The | Goodfellas | You've Got Mail | Sleepless in Seattle | Scent of a Woman |
#                           |:--------------:|:-----------------------:|:----------:|:---------------:|:--------------------:|:----------------:|
#                           |      0.45      |          0.54           |    1.45    |      -0.44      |        -0.45         |      -1.42       |
#                           |      1.47      |          1.48           |    0.48    |      -1.58      |        -1.46         |      -0.54       |
#                           |      1.51      |          1.55           |    0.40    |      -1.46      |        -1.50         |      -0.51       |
#                           |      0.08      |          0.00           |   -0.02    |      -0.02      |         0.07         |      -0.03       |
#                           |     -0.06      |          0.04           |    0.07    |      0.06       |        -0.01         |       0.03       |
#                           |      0.58      |          0.60           |   -0.53    |      -0.47      |        -0.55         |       0.45       |
#                           |      0.59      |          0.48           |   -0.50    |      -0.44      |        -0.59         |       0.50       |
#                           |      0.53      |          0.56           |   -0.48    |      -0.49      |        -0.58         |       0.55       |
#                           |     -0.97      |          -0.91          |   -1.00    |      1.01       |         0.96         |       0.92       |
#                           |     -1.59      |          -1.56          |   -0.56    |      1.56       |         1.50         |       0.58       |
#                           |     -1.56      |          -1.47          |   -0.43    |      1.40       |         1.53         |       0.47       |
#                           |     -1.56      |          -1.57          |   -0.47    |      1.50       |         1.48         |       0.57       |

#This part of the code shows correlation of matrix X
cor(X) %>% knitr::kable(align="c")
#Code results are following:|                        | Godfather, The | Godfather: Part II, The | Goodfellas | You've Got Mail | Sleepless in Seattle | Scent of a Woman |
#                           |:-----------------------|:--------------:|:-----------------------:|:----------:|:---------------:|:--------------------:|:----------------:|
#                           |Godfather, The          |      1.00      |          1.00           |    0.53    |      -1.00      |        -1.00         |      -0.57       |
#                           |Godfather: Part II, The |      1.00      |          1.00           |    0.55    |      -1.00      |        -1.00         |      -0.59       |
#                           |Goodfellas              |      0.53      |          0.55           |    1.00    |      -0.55      |        -0.53         |      -0.99       |
#                           |You've Got Mail         |     -1.00      |          -1.00          |   -0.55    |      1.00       |         1.00         |       0.60       |
#                           |Sleepless in Seattle    |     -1.00      |          -1.00          |   -0.53    |      1.00       |         1.00         |       0.57       |
#                           |Scent of a Woman        |     -0.57      |          -0.59          |   -0.99    |      0.60       |         0.57         |       1.00       |


#This part of the code shows transposed matrix Q
t(Q) %>% knitr::kable(align="c")
#Code result is following:| Godfather, The | Godfather: Part II, The | Goodfellas | You've Got Mail | Sleepless in Seattle | Scent of a Woman |
#                         |:--------------:|:-----------------------:|:----------:|:---------------:|:--------------------:|:----------------:|
#                         |       1        |            1            |     1      |       -1        |          -1          |        -1        |
#                         |       1        |            1            |     -1     |       -1        |          -1          |        1         |

#This part of the code shows matrix P, transposed for viewing purpose
t(P) %>% knitr::kable(align="c")
#Code result is following:|  1   |  2  |  3  | 4 | 5 |  6  |  7  |  8  | 9  |  10  |  11  |  12  |
#                         |:----:|:---:|:---:|:-:|:-:|:---:|:---:|:---:|:--:|:----:|:----:|:----:|
#                         | 1.0  | 1.0 | 1.0 | 0 | 0 | 0.0 | 0.0 | 0.0 | -1 | -1.0 | -1.0 | -1.0 |
#                         | -0.5 | 0.5 | 0.5 | 0 | 0 | 0.5 | 0.5 | 0.5 | 0  | -0.5 | -0.5 | -0.5 |

#This part of the code shows correlation between 
#"Godfather, The","Godfather: Part II, The","Goodfellas",
#"You've Got Mail","Sleepless in Seattle","Scent of a Woman"
#movie ratings 
six_movies <- c(m_1, m_2, m_3, m_4, m_5, m_6)
tmp <- y[,six_movies]
cor(tmp, use="pairwise.complete") %>% knitr::kable(align="c")
#Code results are following:|                        | Godfather, The | Godfather: Part II, The | Goodfellas | You've Got Mail | Sleepless in Seattle | Scent of a Woman |
#                           |:-----------------------|:--------------:|:-----------------------:|:----------:|:---------------:|:--------------------:|:----------------:|
#                           |Godfather, The          |      1.00      |          0.83           |    0.45    |      -0.45      |        -0.35         |       0.07       |
#                           |Godfather: Part II, The |      0.83      |          1.00           |    0.54    |      -0.34      |        -0.33         |       0.14       |
#                           |Goodfellas              |      0.45      |          0.54           |    1.00    |      -0.49      |        -0.37         |      -0.17       |
#                           |You've Got Mail         |     -0.45      |          -0.34          |   -0.49    |      1.00       |         0.54         |      -0.20       |
#                           |Sleepless in Seattle    |     -0.35      |          -0.33          |   -0.37    |      0.54       |         1.00         |      -0.18       |
#                           |Scent of a Woman        |      0.07      |          0.14           |   -0.17    |      -0.20      |        -0.18         |       1.00       |

#### Commented by Khaliun.B 2021.04.14
#### This part of the code was copied from Machine Learning course
#### Course/Section 6: Model Fitting and Recommendation Systems/6.3: Regularization
#### SVD and PCA: Demonstrates principal component analysis (PCA) or singular value decomposition (SVD).

#This part of the code fills N/As with 0 in matrix y
y[is.na(y)] <- 0

#This part of the code subtracts matrix y's rowmeans from itself
y <- sweep(y, 1, rowMeans(y))

#This part of the code performs Principal Component Analysis
#on matrix y and stores results in matrix "pca"
pca <- prcomp(y)
#Code result for class(pca): [1] "prcomp"

#This part of the code diplays dimensions for pca$rotation and then pca$x
dim(pca$rotation)
dim(pca$x)
#Code result for dim(pca$rotation): [1] 454 292
#Code result for dim(pca$x): [1] 292 292

#This part of the code plots standard deviation of pca analysis result pca$sdev
plot(pca$sdev)
#Resulting plot will be included in Final report

#This part of the code creates var_explained variable
#and plots it
var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var_explained)
#Code result for str(var_explained): num [1:292] 0.0364 0.064 0.0868 0.1083 0.1284 ...
#Resulting plot will be included in Final report

library(ggrepel)

#This part of the code creates pcs data frame
#which contains pca$rotation value and columns named as
#"Godfather, The","Godfather: Part II, The","Goodfellas",
#"You've Got Mail","Sleepless in Seattle","Scent of a Woman"
#movies from matrix y
pcs <- data.frame(pca$rotation, name = colnames(y))
str(pcs)
#Code result for dim(pcs):[1] 454 293
#Code results for str(pcs): 'data.frame':	454 obs. of  293 variables:
#                           $ PC1  : num  0.0377 0.0347 0.0157 -0.0165 -0.0248 ...
#                           $ PC2  : num  0.00761 -0.00865 0.00938 0.05186 0.03457 ...
#                           ...

#This part of the code plots data frame pcs
#shows filtered values for PC1 between -0.1 and 0.1
#PC2 between -0.075 and 0.1
pcs %>%  ggplot(aes(PC1, PC2)) + geom_point() + 
  geom_text_repel(aes(PC1, PC2, label=name),
                  data = filter(pcs, 
                                PC1 < -0.1 | PC1 > 0.1 | PC2 < -0.075 | PC2 > 0.1))
#Resulting plot will be included in the Final Report

#This part of the code shows bottom 10 movies ordered by PC1
pcs %>% select(name, PC1) %>% arrange(PC1) %>% slice(1:10) %>% knitr::kable()
#Code results are following:|                          |name                      |   PC1|
#                           |:-------------------------|:-------------------------|-----:|
#                           |Pulp Fiction              |Pulp Fiction              | -0.16|
#                           |Seven (a.k.a. Se7en)      |Seven (a.k.a. Se7en)      | -0.14|
#                           |Fargo                     |Fargo                     | -0.14|
#                           |Taxi Driver               |Taxi Driver               | -0.13|
#                           |2001: A Space Odyssey     |2001: A Space Odyssey     | -0.13|
#                           |Silence of the Lambs, The |Silence of the Lambs, The | -0.13|
#                           |Clockwork Orange, A       |Clockwork Orange, A       | -0.12|
#                           |Being John Malkovich      |Being John Malkovich      | -0.11|
#                           |Fight Club                |Fight Club                | -0.10|
#                           |Godfather, The            |Godfather, The            | -0.10|

#This part of the code shows top 10 movies ordered by PC1
pcs %>% select(name, PC1) %>% arrange(desc(PC1)) %>% slice(1:10) %>% knitr::kable()
#Code results are following: #                                                                                        |name                                                                                    |  PC1|
#                           |:---------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|----:|
#                           |Independence Day (a.k.a. ID4)                                                           |Independence Day (a.k.a. ID4)                                                           | 0.16|
#                           |Shrek                                                                                   |Shrek                                                                                   | 0.13|
#                           |Twister                                                                                 |Twister                                                                                 | 0.12|
#                           |Titanic                                                                                 |Titanic                                                                                 | 0.12|
#                           |Armageddon                                                                              |Armageddon                                                                              | 0.11|
#                           |Spider-Man                                                                              |Spider-Man                                                                              | 0.11|
#                           |Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) |Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) | 0.10|
#                           |Batman Forever                                                                          |Batman Forever                                                                          | 0.10|
#                           |Forrest Gump                                                                            |Forrest Gump                                                                            | 0.10|
#                           |Enemy of the State                                                                      |Enemy of the State                                                                      | 0.09|

#This part of the code shows bottom 10 movies ordered by PC2
pcs %>% select(name, PC2) %>% arrange(PC2) %>% slice(1:10) %>% knitr::kable()
#Code results are following:|                                              |name                                          |   PC2|
#                           |:---------------------------------------------|:---------------------------------------------|-----:|
#                           |Little Miss Sunshine                          |Little Miss Sunshine                          | -0.08|
#                           |Truman Show, The                              |Truman Show, The                              | -0.08|
#                           |Slumdog Millionaire                           |Slumdog Millionaire                           | -0.08|
#                           |Mars Attacks!                                 |Mars Attacks!                                 | -0.07|
#                           |American Beauty                               |American Beauty                               | -0.07|
#                           |Amelie (Fabuleux destin d'Amélie Poulain, Le) |Amelie (Fabuleux destin d'Amélie Poulain, Le) | -0.07|
#                           |City of God (Cidade de Deus)                  |City of God (Cidade de Deus)                  | -0.07|
#                           |Monty Python's Life of Brian                  |Monty Python's Life of Brian                  | -0.07|
#                           |Shawshank Redemption, The                     |Shawshank Redemption, The                     | -0.07|
#                           |Beautiful Mind, A                             |Beautiful Mind, A                             | -0.06|

#This part of the code shows top 10 movies ordered by PC2
pcs %>% select(name, PC2) %>% arrange(desc(PC2)) %>% slice(1:10) %>% knitr::kable()
#Code results are following: |                                                   |name                                               |  PC2|
#                            |:--------------------------------------------------|:--------------------------------------------------|----:|
#                            |Lord of the Rings: The Two Towers, The             |Lord of the Rings: The Two Towers, The             | 0.34|
#                            |Lord of the Rings: The Fellowship of the Ring, The |Lord of the Rings: The Fellowship of the Ring, The | 0.33|
#                            |Lord of the Rings: The Return of the King, The     |Lord of the Rings: The Return of the King, The     | 0.24|
#                            |Matrix, The                                        |Matrix, The                                        | 0.23|
#                            |Star Wars: Episode IV - A New Hope                 |Star Wars: Episode IV - A New Hope                 | 0.22|
#                            |Star Wars: Episode VI - Return of the Jedi         |Star Wars: Episode VI - Return of the Jedi         | 0.19|
#                            |Star Wars: Episode V - The Empire Strikes Back     |Star Wars: Episode V - The Empire Strikes Back     | 0.17|
#                            |Spider-Man 2                                       |Spider-Man 2                                       | 0.11|
#                            |Dark Knight, The                                   |Dark Knight, The                                   | 0.10|
#                            |X2: X-Men United                                   |X2: X-Men United                                   | 0.09|

#### Wrapping up for 2021.04.14 Tested each code
#### Understood the codes for Matrix Factorization
#### Must confess, have no idea what the PCA is about
#### And still haven't figured out what to do with this analysis

#### Commented by Khaliun.B 2021.04.15
#### This part of the code was copied from Textbook Chapter 34 Clustering
#### It demonstrates how to identify groups of movies that are related
#### And it is a part of "Unsupervised machine learning"
#### I can incorporate this into the final model
#1. Perform Hierarchical Clustering algorithm on the data using hclust
#2. Identify groupId for each movieId and add the groupId to the movilens10k table (or edx)
#3. Determine mu's for each group
#4. Incorporate the mu's into the model
#5. Calculate RMSE. It should go down

#This part of the code filters out movieIds of top 50 most rated movies from movielens data
# and assigns the results into data frame named "top"
top <- movielens %>%
  group_by(movieId) %>%
  summarize(n=n(), title = first(title)) %>%
  top_n(50, n) %>%
  pull(movieId)
#Code result for head(top): [1]   1  32  47  50 110 150

#This part of the code filters out users who had rated top 50 most rated movies and
# with total ratings of above 25 (active raters)
x <- movielens %>% 
  filter(movieId %in% top) %>%
  group_by(userId) %>%
  filter(n() >= 25) %>%
  ungroup() %>% 
  select(title, userId, rating) %>%
  spread(userId, rating)
#Code results for head(x):
## A tibble: 6 x 140
#title      `8`  `15`  `17`  `19`  `20`  `21`  `22`  `23`  `26`  `30`  `48`  `56`  `68`  `72`  `73`  `75`  `77`
#<chr>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#  1 Ace Ven…  NA     2    NA       3   1       3  NA     2     0.5     2   3.5    NA  NA    NA     2      NA   3  
#  2 Aladdin   NA     0.5  NA       3   3.5    NA   2     4    NA       5  NA      NA  NA    NA     5      NA   3.5
#  3 America…   4.5   4     4.5    NA  NA      NA   4     3.5   4       5  NA       5  NA     4     4.5    NA   3  
#  4 Apollo …  NA     3    NA       3   3      NA  NA     3.5  NA       5  NA      NA   4     3.5   3.5    NA  NA  
#  5 Back to…   4     5     4.5     5   3.5     4   4     4.5  NA       5   3.5     4   4     3     5       4   3.5
#  6 Batman    NA     4    NA       4   4       3   4.5   3.5  NA       4  NA      NA   3.5  NA     4       1   3.5
#...

#This part of the code populates variable named "row_names"
#with titles that first removed text ": Episode" and cut down to 20 characters
row_names <- str_remove(x$title, ": Episode") %>% str_trunc(20)
#Code results for head(row_names): [1] "Ace Ventura: Pet ..." "Aladdin"              "American Beauty"      "Apollo 13"           
#                                  [5] "Back to the Future"   "Batman"    

#This part of the converts data frame x into matrix and returns its values to itself
#except for the titles for columns
x <- x[,-1] %>% as.matrix()
#Code result for dim(x): [1]  50 139

#This part of the code subtracts column means of matrix x
#from each column and ignores NAs while doing this
x <- sweep(x, 2, colMeans(x, na.rm = TRUE))
#Code results for x[,1]: [1]    NA    NA  0.24    NA -0.26    NA    NA -0.26    NA    NA    NA    NA -0.26 -0.26  0.24  0.74  0.74 -0.26
#[19] -1.26    NA    NA    NA -0.76 -0.26 -0.26    NA  0.74    NA    NA -0.26 -0.26 -0.26 -0.26  0.74  0.74  0.74
#[37]    NA  0.24  0.24    NA -0.76 -0.76 -0.26 -0.26    NA    NA    NA    NA  0.74  0.74

#This part of the code subtracts row means of matrix x
#from each row and ignores NAs while doing this
x <- sweep(x, 1, rowMeans(x, na.rm = TRUE))
#Code result for length(x[1,]): [1] 139
#Code result for rownames(x): NULL

#This part of the code sets row names for matrix x
rownames(x) <- row_names
#Code results for head(rownames(x)): [1] "Ace Ventura: Pet ..." "Aladdin"              "American Beauty"      "Apollo 13"           
#                                    [5] "Back to the Future"   "Batman"

#This part of the code creates "d" of class "dist" from matrix "x". Prepares data for clustering
d <- dist(x)
#Code result for class(d): [1] "dist"
#Code results for str(d): 'dist' num [1:1225] 14.2 15.5 12.3 12.5 11.9 ...
#                         - attr(*, "Size")= int 50
#                         - attr(*, "Labels")= chr [1:50] "Ace Ventura: Pet ..." "Aladdin" "American Beauty" "Apollo 13" ...
#                         - attr(*, "Diag")= logi FALSE
#                         - attr(*, "Upper")= logi FALSE
#                         - attr(*, "method")= chr "euclidean"
#                         - attr(*, "call")= language dist(x = x)

####################################################################################
#### This part of the code was copied from Textbook 34.1 Hierarchical clustering
#### Note: k value should be tuned
####################################################################################

#This part of the code populates variable "h" of "hclust"
# from "d"
h <- hclust(d)
#Code result for class(h): [1] "hclust"
#Code result for str(hclust): function (d, method = "complete", members = NULL)  

#This part of the code creates plot showing a tree of movie groups
#We can see the resulting groups using a dendrogram.
plot(h, cex = 0.65, main = "", xlab = "")
#Resulting plot will be included in final report

#This part of the code populates integer class variable "groups"
#with group ids from "h". Value k should be tuned
groups <- cutree(h, k = 10)
#Code result for class(groups): [1] "integer"
#Code result for str(groups):  Named int [1:50] 1 2 3 4 5 6 2 4 4 1 ...
#                             - attr(*, "names")= chr [1:50] "Ace Ventura: Pet ..." "Aladdin" "American Beauty" "Apollo 13" ...

#This part of the code shows names of movies that belong to group 4
names(groups)[groups==4]
#Code results are following: [1] "Apollo 13"            "Braveheart"           "Dances with Wolves"   "Forrest Gump"        
#                             [5] "Good Will Hunting"    "Saving Private Ryan"  "Schindler's List"     "Shawshank Redempt..."

#This part of the code shows names of movies that belong to group 9
names(groups)[groups==9]
#Code results are following: [1] "Lord of the Rings..." "Lord of the Rings..." "Lord of the Rings..." "Star Wars IV - A ..."
#                            [5] "Star Wars V - The..." "Star Wars VI - Re..."

#We can also explore the data to see if there are clusters of movie raters.
h_2 <- dist(t(x)) %>% hclust()

####################################################################################
#### This part of the code was copied from Textbook 34.2 k-means
####################################################################################

#This part of the code populates matrix "x_0" with data from matrix x
x_0 <- x
dim(x_0)
#Code result for dim(x_0): [1]  50 139

#This part of the code fills NAs of the matrix x_0 with value 0
#The kmeans function included in R-base does not handle NAs. We are using 0's to fill out the NAs
x_0[is.na(x_0)] <- 0
#Code results for head(x_0[,3]): Ace Ventura: Pet ...              Aladdin      American Beauty            Apollo 13   Back to the Future 
#                                 0.00                 0.00                 0.41                 0.00                 0.59 
#                                 Batman 
#                                 0.00 

#This part of the code calculates kmeans and assigns the value to variable "k"
k <- kmeans(x_0, centers = 10)
#Code results class(k): [1] "kmeans"
#Code results for str(k): List of 9
#                         $ cluster     : Named int [1:50] 9 2 3 7 6 4 2 7 7 9 ...
#                         ..- attr(*, "names")= chr [1:50] "Ace Ventura: Pet ..." "Aladdin" "American Beauty" "Apollo 13" ...
#                         $ centers     : num [1:10, 1:139] -0.7098 -0.1285 -0.0766 0.0417 0.2525 ...
#                         ..- attr(*, "dimnames")=List of 2
#                         .. ..$ : chr [1:10] "1" "2" "3" "4" ...
#                         .. ..$ : chr [1:139] "8" "15" "17" "19" ...
#                         $ totss       : num 2837
#                         $ withinss    : num [1:10] 51.7 391.8 166.2 346.7 93.4 ...
#                         $ tot.withinss: num 1783
#                         $ betweenss   : num 1054
#                         $ size        : int [1:10] 3 9 5 8 3 6 9 3 3 1
#                         $ iter        : int 3
#                         $ ifault      : int 0
#                         - attr(*, "class")= chr "kmeans"

#This part of the code assigns group ids calculated by 
#kmeans to "groups" variable
groups <- k$cluster
#Code results for str(groups): Named int [1:50] 9 2 3 7 6 4 2 7 7 9 ...
#                               - attr(*, "names")= chr [1:50] "Ace Ventura: Pet ..." "Aladdin" "American Beauty" "Apollo 13" ...

#This part of the code shows names of movies that belong to group 9
names(groups)[groups==7]
#Code results are following: [1] "Apollo 13"            "Braveheart"           "Dances with Wolves"   "E.T. the Extra-Te..."
#                           [5] "Gladiator"            "Godfather, The"       "Good Will Hunting"    "Saving Private Ryan" 
#                           [9] "Schindler's List"

#This part of the code calculates kmeans (nstart parameter added) and assigns the value to variable "k"
k <- kmeans(x_0, centers = 10, nstart = 25)

#### Wrapping up for 2021.04.15 Tested each code
#### Decided to add Movie groups effect
#### to the Final model to improve RMSE