###########################################################
# This project is based on adult census income datasets
# The goal is to predict income (> 50K or <50K) 
###########################################################

# Download packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

# Download datasets: adult.data, adult.test, adult.names
train_url <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data' 
test_url <-'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test' 
name_url <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names' 

adult_names <- readLines(name_url)[97:110]
adult_names <- as.character(lapply(strsplit(adult_names,':'), function(x) x[1])) 
adult_names <- c(adult_names,'income')

# create temporate train and test sets (with "NA")
train_temp <- read.table(train_url,sep = ',', col.names = adult_names, 
                         na.strings = ' ?', stringsAsFactors = TRUE)
test_temp <- readLines(test_url)[-1]
test_temp <- read.table(textConnection(test_temp),sep = ',',
                        col.names = adult_names, 
                        na.strings = ' ?',stringsAsFactors = TRUE)
# read variables
str(train_temp)
# check the proportion of "NA" in train_temp and test_temp
train_NA <- 1 - mean(complete.cases(train_temp))
test_NA <- 1 - mean(complete.cases(test_temp))

# clean data: remove "NA" in train_temp and test_temp
train_set <- train_temp[!is.na(train_temp$workclass) 
                        & !is.na(train_temp$occupation) &
                          !is.na(train_temp$native.country),]
test_set <- test_temp[!is.na(test_temp$workclass) 
                      & !is.na(test_temp$occupation) &
                        !is.na(test_temp$native.country),]

#To make train_set and test_set consistent with each other,
# change " >50K." and " <=50K." in test_set to " >50K" and " <=50K"
test_set$income <- as.character(test_set$income)
for (i in 1:nrow(test_set))
  if (test_set$income[i]==" >50K."){
    test_set$income[i] <- " >50K"
  }else
  {
    test_set$income[i] <- " <=50K"
  }
test_set$income <- as.factor(test_set$income)

# count the observation number of train_set and test_set
train_nrow <- nrow(train_set)
test_nrow <- nrow(test_set)

###########################################################
# Visualizaion
###########################################################

# number of individuals in different income levels
train_set %>% group_by(income) %>% summarize(n=n())

# age distribution for different income levels
boxplot (age ~ income, data = train_set, 
         main = "Age distribution for different income levels",
         xlab = "Income level", ylab = "Age", col = "blue")

# education years distribution for different income levels
boxplot (education.num ~ income, data = train_set, 
         main = "Education years distribution for different income levels",
         xlab = "Income level", ylab = "Education year", col = "blue")

# capital loss distribution for two income levels
boxplot (capital.loss ~ income, data = train_set, 
         main = "Capital loss distribution for different income levels",
         xlab = "Income level", ylab = "Capital loss", col = "blue")

# capital gain for two income levels
boxplot (capital.gain ~ income, data = train_set, 
         main = "Capital gain distribution for different income levels",
         xlab = "Income level", ylab = "Capital gain", col = "blue")

# ratio when capital gain equals to 0
mean(train_set$capital.gain==0)
# ratio when capital loss equals to 0
mean(train_set$capital.loss==0)

# working time distribution for two income levels
boxplot (hours.per.week ~ income, data = train_set, 
         main = "Working time for different income levels",
         xlab = "Income level", ylab = "Hours per week", col = "blue")

cor (
  train_set[, c("age", "education.num", "hours.per.week")])

# table with sex versus income levels
train_set %>% group_by(sex) %>% summarize(lower=mean(income==" <=50K"), higher = mean(income==" >50K"))

# barplot of each relationship for two income levels
qplot (income, data = train_set, 
       fill = relationship) + facet_grid (. ~ relationship)

# barplot of each race for two income levels
qplot (income, data = train_set, 
       fill = race) + facet_grid (. ~ race)

# barplot of each marital status for two income levels
qplot (income, data = train_set, 
       fill = marital.status) + facet_grid (. ~ marital.status)

# barplot of each workclass for two income levels
qplot (income, data = train_set, 
       fill = workclass) + facet_grid (. ~ workclass)

# barplot of each occupation for two income levels
qplot (income, data = train_set, 
       fill = occupation) + facet_grid (. ~ occupation)

# ratio for native country is United states
mean(train_set$native.country==" United-States")

########################################################
# Prediction model
########################################################

# remove non-predictor variables in train_set
train_set <- train_set %>% select(-fnlwgt, -capital.gain, -capital.loss, - education, - native.country) 

# divide train_set into learning set and validation set
set.seed(100)
valid_index <- createDataPartition(y = train_set$income, times = 1, p = 0.1, list = FALSE)
learning <- train_set[-valid_index,]
validation <- train_set[valid_index,]

###########################################################
# train with glm model
train_glm <- train(income ~ age + education.num + hours.per.week + sex + relationship + marital.status + occupation, method = "glm", data = learning)
# predict using validation set
glm_valid <- predict(train_glm, validation)
# confusion matrix
glm_confM <- confusionMatrix(data=validation$income, reference=glm_valid) 

###########################################################
# train with decision tree model
set.seed(10)
train_rpart <- train(income ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.02, 0.001)),
                     data = learning)
# plot accuracy versus compexity parameter (cp)
ggplot(train_rpart)

# observe our final tree 
learning_tree <- train_rpart$finalModel

# predict using validation set and confusion matrix
rpart_valid <- predict(train_rpart, validation)
rpart_confM <- confusionMatrix(data = validation$income, reference = rpart_valid)

###########################################################
# train with SVM model
train_svm <- svm(income ~.,data=learning)
# predict with validation set and confusion matrix
svm_valid <- predict(train_svm,validation)
svm_confM <- confusionMatrix(data=validation$income, reference=svm_valid)

###########################################################
# Result
###########################################################

# remove non predictor variables from test_set
test_set <- test_set%>% select(-fnlwgt,-capital.gain,-capital.loss, -native.country, - education)

# glm test results for test_set
glm_test <- predict(train_glm, test_set)
glm_test_confM <- confusionMatrix(data=test_set$income, reference=glm_test)

# decision tree test results for test_set
rpart_test <- predict(train_rpart, test_set)
rpart_test_confM <- confusionMatrix(data = test_set$income, reference = rpart_test)

# svm test results for test_set
svm_test <- predict(train_svm, test_set)
svm_test_confM <- confusionMatrix(data=test_set$income, reference=svm_test)

# Done
############################################################