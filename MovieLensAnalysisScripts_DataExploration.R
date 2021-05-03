#### Data exloration scripts and answers
#### Tested by Khaliun.B 2021.04.05-2021.04.11
#### According to the MovieLens project instructions
#### (!) First, there will be a short quiz on the MovieLens data. You can view this quiz as an opportunity to familiarize yourself with the data in order to prepare for your project submission.

source("MovieLensAnalysisScripts.R", local = knitr::knit_global())

#Q1: How many rows and columns are there in the edx dataset?
dim(edx)
# Both answers hold out: [1] 9000055       6
#Q2-1: How many zeros were given as ratings in the edx dataset?
edx %>% filter(rating == 0) %>% tally()
# Answer holds out: 0
#Q2-2: How many threes were given as ratings in the edx dataset?
edx %>% filter(rating == 3) %>% tally()
# Answer holds out: 2121240
#Q3: How many different movies are in the edx dataset?
n_distinct(edx$movieId)
# Answer holds out: [1] 10677
#Q4: How many different users are in the edx dataset?
n_distinct(edx$userId)
# Answer holds out: [1] 69878
#Q5: How many movie ratings are in each of the following genres in the edx dataset?
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})
#edx %>% separate_rows(genres, sep = "\\|") %>% # Not using this part of the code
#  group_by(genres) %>%
#  summarize(count = n()) %>%
#  arrange(desc(count))
#Answers hold out: 
#Drama   Comedy Thriller  Romance 
#3910127  3540930  2325899  1712100 

#Q6: Which movie has the greatest number of ratings?
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
#Answer holds out:
#movieId title count
#<dbl> <chr> <int>
#  1   296 Pulp Fiction (1994) 31362

#Q7: What are the five most given ratings in order from most to least?
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))
#Answer holds out:
#rating   count
#<dbl>   <int>
#1    4   2588430
#2    3   2121240
#3    5   1390114
#4    3.5  791624
#5    2    711422

#Q8: True or False: In general, half star ratings are less common than whole star ratings (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()
#Answer holds out: True

#Added by Khaliun.B 2021.04.11
#Checking if validation set holds the 90:10 proportion for edx:validation data as stated in initial script provided by Edx (please see file: MovieLensAnalysisScript.R)
dim(validation)
#Answer: [1] 999999      6
#Validation set hold 999999 rows and 6 columns Which is aproximately 11% of edx set 9000055 and 10% of initial data

#Checking structure and of edx set: Commented by Khaliun.B 2021.04.11
str(edx)
#Classes ‘data.table’ and 'data.frame':	9000055 obs. of  6 variables: Commented by Khaliun.B 2021.04.11"
# columns: userId[int] movieId[num] rating[num] timestamp[int] title[chr] genres[chr]: Commented by Khaliun.B 2021.04.11

#Checking structure and of validation set: Commented by Khaliun.B 2021.04.11
str(validation)
# Classes ‘data.table’ and 'data.frame':	999999 obs. of  6 variables: Commented by Khaliun.B 2021.04.11"
# columns: userId[int] movieId[num] rating[num] timestamp[int] title[chr] genres[chr]: Commented by Khaliun.B 2021.04.11"


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
#   movies_kmeans <- movielens %>%
#     group_by(movieId) %>%
#     summarize(n=n(), title = first(title)) %>%
#     filter(n>=50) %>%
#     pull(movieId)
#Code result for head(top): [1]   1  32  47  50 110 150

#This part of the code filters out users who had rated top 50 most rated movies and
# with total ratings of above 25 (active raters): Commented by Khaliun.B 2021.04.15
#   x <- movielens %>% 
#     filter(movieId %in% movies_kmeans) %>%
#     group_by(userId) %>%
#     filter(n() >= 50) %>%
#     ungroup() %>% 
#     select(movieId,title, userId, rating) %>%
#     spread(userId, rating)
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
#   row_names <- x$title
#   row_movieIds <- x$movieId
#Code results for head(row_names): [1] "Ace Ventura: Pet ..." "Aladdin"              "American Beauty"      "Apollo 13"           
#                                  [5] "Back to the Future"   "Batman"    

#This part of the converts data frame x into matrix and returns its values to itself
#except for the titles for columns: Commented by Khaliun.B 2021.04.15
#   x_0 <- x[,-1]
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  |               title                | 8  | 15 | 17  | 19 | 20  | 21 | 22  | 23  | 26  |
#  |:----------------------------------:|:--:|:--:|:---:|:--:|:---:|:--:|:---:|:---:|:---:|
#  |             Toy Story              | NA | 2  | NA  | 3  | 3.5 | NA | NA  | 3.0 | 5.0 |
#  | Twelve Monkeys (a.k.a. 12 Monkeys) | 5  | 4  | 4.5 | 3  | 2.5 | 4  | 4.5 | 4.0 | 4.5 |
#  |        Seven (a.k.a. Se7en)        | 5  | 5  | 5.0 | 5  | NA  | 4  | 3.5 | 4.5 | 4.5 |
#  |        Usual Suspects, The         | 5  | 5  | 5.0 | 4  | NA  | NA | NA  | 4.0 | 4.5 |

#   x_0 <- x_0[,-1] %>% as.matrix()
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  | 8  | 15 | 17  | 19 | 20  | 21 | 22  | 23  | 26  | 30 |
#  |:--:|:--:|:---:|:--:|:---:|:--:|:---:|:---:|:---:|:--:|
#  | NA | 2  | NA  | 3  | 3.5 | NA | NA  | 3.0 | 5.0 | 4  |
#  | 5  | 4  | 4.5 | 3  | 2.5 | 4  | 4.5 | 4.0 | 4.5 | 2  |
#  | 5  | 5  | 5.0 | 5  | NA  | 4  | 3.5 | 4.5 | 4.5 | 4  |
#  | 5  | 5  | 5.0 | 4  | NA  | NA | NA  | 4.0 | 4.5 | 5  |
#Code results for dim(x_0):
#   dim(x_0)

#This part of the code subtracts column means of matrix x
#from each column and ignores NAs while doing this: Commented by Khaliun.B 2021.04.15
#   x_0 <- sweep(x_0, 2, colMeans(x_0, na.rm = TRUE))
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
# |  8   |  15   |  17  |  19   |  20   |  21  |  22   |  23  |  26  |  30   |
# |:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:----:|:----:|:-----:|
# |  NA  | -1.52 |  NA  | -0.78 | 0.67  |  NA  |  NA   | -1.0 | 1.45 | -0.27 |
# | 0.74 | 0.48  | 0.71 | -0.78 | -0.33 | 0.31 | 0.42  | 0.0  | 0.95 | -2.27 |
# | 0.74 | 1.48  | 1.21 | 1.22  |  NA   | 0.31 | -0.58 | 0.5  | 0.95 | -0.27 |
# | 0.74 | 1.48  | 1.21 | 0.22  |  NA   |  NA  |  NA   | 0.0  | 0.95 | 0.73  |

#This part of the code subtracts row means of matrix x
#from each row and ignores NAs while doing this: Commented by Khaliun.B 2021.04.15
#   x_0 <- sweep(x_0, 1, rowMeans(x_0, na.rm = TRUE))
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  |  8   |  15   |  17  |  19   |  20   |  21  |  22   |  23   |  26  |  30   |
#  |:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|
#  |  NA  | -1.48 |  NA  | -0.74 | 0.71  |  NA  |  NA   | -0.96 | 1.49 | -0.23 |
#  | 0.71 | 0.45  | 0.68 | -0.81 | -0.36 | 0.28 | 0.39  | -0.03 | 0.92 | -2.30 |
#  | 0.54 | 1.28  | 1.01 | 1.02  |  NA   | 0.11 | -0.78 | 0.30  | 0.75 | -0.47 |
#  | 0.33 | 1.07  | 0.80 | -0.19 |  NA   |  NA  |  NA   | -0.41 | 0.54 | 0.32  |

#This part of the code sets row names for matrix x: Commented by Khaliun.B 2021.04.15
#Currently using titles as row names for illustrative purposes to view titles in heatmap
#   rownames(x_0) <- row_names
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  |                     |  8   |  15   |  17  |  19   |  20   |  21  |  22   |  23   |  26  |  30   |
#  |:--------------------|:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|
#  |Toy Story            |  NA  | -1.48 |  NA  | -0.74 | 0.71  |  NA  |  NA   | -0.96 | 1.49 | -0.23 |
#  |Twelve Monkeys (a... | 0.71 | 0.45  | 0.68 | -0.81 | -0.36 | 0.28 | 0.39  | -0.03 | 0.92 | -2.30 |
#  |Seven (a.k.a. Se7en) | 0.54 | 1.28  | 1.01 | 1.02  |  NA   | 0.11 | -0.78 | 0.30  | 0.75 | -0.47 |
#  |Usual Suspects, The  | 0.33 | 1.07  | 0.80 | -0.19 |  NA   |  NA  |  NA   | -0.41 | 0.54 | 0.32  |

#This part of the code fills NAs of the matrix x_0 with value 0: Commented by Khaliun.B 2021.04.15
#The kmeans function included in R-base does not handle NAs. We are using 0's to fill out the NAs: Explanatory text from Textbook
#   x_0[is.na(x_0)] <- 0
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):

#This part of the code reassigns rownames as movieIds as a preparation to mutate the groups back
#to original data frame
#   rownames(x_0) <- row_movieIds
#Code results for x_0[1:4,1:10]%>%knitr::kable(align="c"):
#  |   |  8   |  15   |  17  |  19   |  20   |  21  |  22   |  23   |  26  |  30   |
#  |:--|:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|
#  |1  | 0.00 | -1.48 | 0.00 | -0.74 | 0.71  | 0.00 | 0.00  | -0.96 | 1.49 | -0.23 |
#  |32 | 0.71 | 0.45  | 0.68 | -0.81 | -0.36 | 0.28 | 0.39  | -0.03 | 0.92 | -2.30 |
#  |47 | 0.54 | 1.28  | 1.01 | 1.02  | 0.00  | 0.11 | -0.78 | 0.30  | 0.75 | -0.47 |
#  |50 | 0.33 | 1.07  | 0.80 | -0.19 | 0.00  | 0.00 | 0.00  | -0.41 | 0.54 | 0.32  |

#This part of the code recalculates k mutate the groups back to original data frame
#   k <- kmeans(x_0, centers = 10, nstart=25)
#   summary(k)

#This part of the code assigns group ids calculated by 
#kmeans to "groups" variable: Commented by Khaliun.B 2021.04.15
#   groups <- factor(k$cluster)
#Code results for summary(groups)
#Code results for length(names(groups))

#   temp_g<-data.frame(movieId=as.integer(names(groups)))
#Checking if the length of movieId is the same as the factor names()
#   length(temp_g$movieId)
#Code results:

#Mutate group numbers back to the dataset
#   temp_g<-temp_g%>%mutate(mgroup=as.integer(groups[names(groups)==.$movieId]))
#Code results for head(temp_g):

#Checking if the groups had been assigned the right way
#   temp_g%>%filter(mgroup==7)
#   names(groups)[groups==7]

#Mutate groups back to original data
#   movielens <- movielens %>% 
#     left_join(temp_g, by='movieId')

#Checking the resulting data frame
#   summary(x_mur_g)

#x_mur_g%>%filter(mgroup==15)%>%group_by(movieId,title,kgroup)%>%summarize(n=n(),mu=mean(rating))%>%select(movieId,title,mu,n,kgroup)

################################################################################
#### END: This group of code performs kmeans clustering for movielens data
#### Commented by Khaliun.B 2021.04.18
################################################################################

################################################################################
#### BEGIN: This group of code performs lambda tuning for sample edx data
#### Uses cross-validation: Commented by Khaliun.B 2021.05.03
################################################################################

#This part of the code creates sequence of lamdba values for parameter tuning: Commented by Khaliun.B 2021.04.26
lambdas <- seq(0, 10, 0.25)
#Code result for head(lambdas): [1] 0.00 0.25 0.50 0.75 1.00 1.25

#This part of the code creates function for calculating Residual Mean Squared Error: Commented by Khaliun.B 2021.04.26
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
    left_join(b_r, by = "rating") %>%
    mutate(pred = mu + b_i + b_u + b_r) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

#This part of the code creates qplot of lambdas against rmses: Commented by Khaliun.B 2021.04.26
qplot(lambdas, rmses2)
#Code result shows in a qplot in Plots tab  which will be included in final report

#This part of the code shows value of lambda correspoding to minimum value of rmses
#and stores the minimum value in variable "lambda": Commented by Khaliun.B 2021.04.26
lambda <- lambdas[which.min(rmses2)]
lambda
#Code result is following: [1] 4.75

#This part of the code creates data frame "rmse_results" which contains RMSE value
# Regularized Movie+User Effect Model RMSE: Commented by Khaliun.B 2021.04.26
rmse_results <-data_frame(method="Regularized Movie + User Effect Model",
                          RMSE = min(rmses))
rmse_results %>% knitr::kable()

################################################################################
#### END: This group of code performs lambda tuning for sample edx data
#### Commented by Khaliun.B 2021.05.03
################################################################################