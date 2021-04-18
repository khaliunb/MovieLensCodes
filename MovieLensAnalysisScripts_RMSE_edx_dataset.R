#### RMSE calculation script for MovieLens project
#### This file holds the script that is based on the edx data set and will not use final validation data set
#### Created by Khaliun.B 2021.04.11
#### According to the MovieLens project instructions
#### (!) Second, you will train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set.
#### 25 points: RMSE < 0.86490

#### The final model follows the Regularized Movie + User Effects Model: Yu,i=μ+bi+bu+ϵu,i

#This part of the code added by Khaliun.B 2021.04.17. library factoextra is needed
#for fviz_cluster() function for creating cluster plots in
#file: MovieLensAnalysisScripts_RMSE_edx_dataset.R
#if(!require(factoextra)) install.packages("factoextra", repos = "http://cran.us.r-project.org")
#

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

################################################################################
#### BEGIN: This group of code performs kmeans clustering for movielens data
#### Commented by Khaliun.B 2021.04.18
################################################################################

#### Commented by Khaliun.B 2021.04.15
#### This part of the code was copied from Textbook Chapter 34 Clustering
#### It demonstrates how to identify groups of movies that are related
#### And it is a part of "Unsupervised machine learning"
#### I can incorporate this into the final model
#- Identify groupId for each movieId and add the groupId to the movilens10k table (or edx)
#- Determine mu's for each group
#- Incorporate the mu's into the model
#- Calculate RMSE. It should go down

####################################################################################
#### Creating k-means clusters
####################################################################################

#This part of the code filters out movieIds that were rated at least 25 times from movielens data
# and assigns the results into data frame named "top": Commented by Khaliun.B 2021.04.15
movies_kmeans <- movielens %>%
  group_by(movieId) %>%
  summarize(n=n(), title = first(title)) %>%
  filter(n>=50) %>%
  pull(movieId)
#Code result for head(top): [1]   1  32  47  50 110 150

#This part of the code filters out users who had rated top 50 most rated movies and
# with total ratings of above 25 (active raters): Commented by Khaliun.B 2021.04.15
x <- movielens %>% 
  filter(movieId %in% movies_kmeans) %>%
  group_by(userId) %>%
  filter(n() >= 50) %>%
  ungroup() %>% 
  select(movieId,title, userId, rating) %>%
  spread(userId, rating)
#Code results for head(x):
#movieId title    `8`  `15`  `17`  `19`  `20`  `21`  `22`  `23`  `26`  `30`  `48`  `56`  `68`  `72`  `73`  `75`
#<int> <chr>  <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#  1       1 Toy S…    NA     2  NA       3   3.5    NA  NA     3     5       4     4     4     4   3.5   5       3
#  2      32 Twelv…     5     4   4.5     3   2.5     4   4.5   4     4.5     2    NA    NA    NA  NA     5       4
#  3      47 Seven…     5     5   5       5  NA       4   3.5   4.5   4.5     4    NA     4    NA   3.5   5      NA
#  4      50 Usual…     5     5   5       4  NA      NA  NA     4     4.5     5    NA     4    NA   4     5      NA
#  5     110 Brave…     4     3  NA       3   2      NA  NA     3.5  NA       5     4    NA    NA   3.5   4      NA
#  6     150 Apoll…    NA     3  NA       3   3      NA  NA     3.5  NA       5    NA    NA     4   3.5   3.5    NA
# ...

#This part of the code populates variable named "row_names"
#with titles that first removed text ": Episode" and cut down to 20 characters: Commented by Khaliun.B 2021.04.15
row_names <- x$title
row_movieIds <- x$movieId
#Code results for head(row_names): [1] "Ace Ventura: Pet ..." "Aladdin"              "American Beauty"      "Apollo 13"           
#                                  [5] "Back to the Future"   "Batman"    

#This part of the converts data frame x into matrix and returns its values to itself
#except for the titles for columns: Commented by Khaliun.B 2021.04.15
x_0 <- x[,-1]
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  |               title                | 8  | 15 | 17  | 19 | 20  | 21 | 22  | 23  | 26  |
#  |:----------------------------------:|:--:|:--:|:---:|:--:|:---:|:--:|:---:|:---:|:---:|
#  |             Toy Story              | NA | 2  | NA  | 3  | 3.5 | NA | NA  | 3.0 | 5.0 |
#  | Twelve Monkeys (a.k.a. 12 Monkeys) | 5  | 4  | 4.5 | 3  | 2.5 | 4  | 4.5 | 4.0 | 4.5 |
#  |        Seven (a.k.a. Se7en)        | 5  | 5  | 5.0 | 5  | NA  | 4  | 3.5 | 4.5 | 4.5 |
#  |        Usual Suspects, The         | 5  | 5  | 5.0 | 4  | NA  | NA | NA  | 4.0 | 4.5 |

x_0 <- x_0[,-1] %>% as.matrix()
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  | 8  | 15 | 17  | 19 | 20  | 21 | 22  | 23  | 26  | 30 |
#  |:--:|:--:|:---:|:--:|:---:|:--:|:---:|:---:|:---:|:--:|
#  | NA | 2  | NA  | 3  | 3.5 | NA | NA  | 3.0 | 5.0 | 4  |
#  | 5  | 4  | 4.5 | 3  | 2.5 | 4  | 4.5 | 4.0 | 4.5 | 2  |
#  | 5  | 5  | 5.0 | 5  | NA  | 4  | 3.5 | 4.5 | 4.5 | 4  |
#  | 5  | 5  | 5.0 | 4  | NA  | NA | NA  | 4.0 | 4.5 | 5  |
#Code results for dim(x_0):
dim(x_0)

#This part of the code subtracts column means of matrix x
#from each column and ignores NAs while doing this: Commented by Khaliun.B 2021.04.15
x_0 <- sweep(x_0, 2, colMeans(x_0, na.rm = TRUE))
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
# |  8   |  15   |  17  |  19   |  20   |  21  |  22   |  23  |  26  |  30   |
# |:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:----:|:----:|:-----:|
# |  NA  | -1.52 |  NA  | -0.78 | 0.67  |  NA  |  NA   | -1.0 | 1.45 | -0.27 |
# | 0.74 | 0.48  | 0.71 | -0.78 | -0.33 | 0.31 | 0.42  | 0.0  | 0.95 | -2.27 |
# | 0.74 | 1.48  | 1.21 | 1.22  |  NA   | 0.31 | -0.58 | 0.5  | 0.95 | -0.27 |
# | 0.74 | 1.48  | 1.21 | 0.22  |  NA   |  NA  |  NA   | 0.0  | 0.95 | 0.73  |

#This part of the code subtracts row means of matrix x
#from each row and ignores NAs while doing this: Commented by Khaliun.B 2021.04.15
x_0 <- sweep(x_0, 1, rowMeans(x_0, na.rm = TRUE))
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  |  8   |  15   |  17  |  19   |  20   |  21  |  22   |  23   |  26  |  30   |
#  |:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|
#  |  NA  | -1.48 |  NA  | -0.74 | 0.71  |  NA  |  NA   | -0.96 | 1.49 | -0.23 |
#  | 0.71 | 0.45  | 0.68 | -0.81 | -0.36 | 0.28 | 0.39  | -0.03 | 0.92 | -2.30 |
#  | 0.54 | 1.28  | 1.01 | 1.02  |  NA   | 0.11 | -0.78 | 0.30  | 0.75 | -0.47 |
#  | 0.33 | 1.07  | 0.80 | -0.19 |  NA   |  NA  |  NA   | -0.41 | 0.54 | 0.32  |

#This part of the code sets row names for matrix x: Commented by Khaliun.B 2021.04.15
#Currently using titles as row names for illustrative purposes to view titles in heatmap
rownames(x_0) <- row_names
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  |                     |  8   |  15   |  17  |  19   |  20   |  21  |  22   |  23   |  26  |  30   |
#  |:--------------------|:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|
#  |Toy Story            |  NA  | -1.48 |  NA  | -0.74 | 0.71  |  NA  |  NA   | -0.96 | 1.49 | -0.23 |
#  |Twelve Monkeys (a... | 0.71 | 0.45  | 0.68 | -0.81 | -0.36 | 0.28 | 0.39  | -0.03 | 0.92 | -2.30 |
#  |Seven (a.k.a. Se7en) | 0.54 | 1.28  | 1.01 | 1.02  |  NA   | 0.11 | -0.78 | 0.30  | 0.75 | -0.47 |
#  |Usual Suspects, The  | 0.33 | 1.07  | 0.80 | -0.19 |  NA   |  NA  |  NA   | -0.41 | 0.54 | 0.32  |

#This part of the code fills NAs of the matrix x_0 with value 0: Commented by Khaliun.B 2021.04.15
#The kmeans function included in R-base does not handle NAs. We are using 0's to fill out the NAs: Explanatory text from Textbook
x_0[is.na(x_0)] <- 0
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):

#This part of the code reassigns rownames as movieIds as a preparation to mutate the groups back
#to original data frame
rownames(x_0) <- row_movieIds
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  |   |  8   |  15   |  17  |  19   |  20   |  21  |  22   |  23   |  26  |  30   |
#  |:--|:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|
#  |1  | 0.00 | -1.48 | 0.00 | -0.74 | 0.71  | 0.00 | 0.00  | -0.96 | 1.49 | -0.23 |
#  |32 | 0.71 | 0.45  | 0.68 | -0.81 | -0.36 | 0.28 | 0.39  | -0.03 | 0.92 | -2.30 |
#  |47 | 0.54 | 1.28  | 1.01 | 1.02  | 0.00  | 0.11 | -0.78 | 0.30  | 0.75 | -0.47 |
#  |50 | 0.33 | 1.07  | 0.80 | -0.19 | 0.00  | 0.00 | 0.00  | -0.41 | 0.54 | 0.32  |

#This part of the code recalculates k mutate the groups back to original data frame
k <- kmeans(x_0, centers = 10, nstart=25)
summary(k)

#This part of the code assigns group ids calculated by 
#kmeans to "groups" variable: Commented by Khaliun.B 2021.04.15
groups <- factor(k$cluster)
#Code results for summary(groups)
#Code results for length(names(groups))

temp_g<-data.frame(movieId=as.integer(names(groups)))
#Checking if the length of movieId is the same as the factor names()
length(temp_g$movieId)
#Code results:

#Mutate group numbers back to the dataset
temp_g<-temp_g%>%mutate(mgroup=as.integer(groups[names(groups)==.$movieId]))
#Code results for head(temp_g):

#Checking if the groups had been assigned the right way
temp_g%>%filter(mgroup==7)
names(groups)[groups==7]

#Mutate groups back to original data
movielens <- movielens %>% 
  left_join(temp_g, by='movieId')

#Checking the resulting data frame
summary(x_mur_g)

#x_mur_g%>%filter(mgroup==15)%>%group_by(movieId,title,kgroup)%>%summarize(n=n(),mu=mean(rating))%>%select(movieId,title,mu,n,kgroup)

################################################################################
#### END: This group of code performs kmeans clustering for movielens data
#### Commented by Khaliun.B 2021.04.18
################################################################################

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
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "mgroup")
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
  b_gi <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(mgroup) %>%
    summarize(b_gi = sum(rating - b_i - b_u - mu)/(n()+l))
  #Code results for summary(movie_group_avgs):
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_gi, by = "mgroup")%>%
    mutate(pred = mu + b_i + b_u + b_gi) %>%
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
                          data_frame(method="Regularized Movie + User Effect Model for Lambda",
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()