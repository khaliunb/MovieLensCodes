#### Data exloration scripts and answers
#### Tested by Khaliun.B 2021.04.05-2021.04.11
#### According to the MovieLens project instructions
#### (!) First, there will be a short quiz on the MovieLens data. You can view this quiz as an opportunity to familiarize yourself with the data in order to prepare for your project submission.

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