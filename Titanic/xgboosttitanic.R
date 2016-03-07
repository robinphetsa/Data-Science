library(xgboost)
library(rpart)
library(Ckmeans.1d.dp)

set.seed(415)

train <- read.csv("../input/train.csv")
test <- read.csv("../input/test.csv")

feature_eng <- function(train_df, test_df) {
  # Combining the train and test sets for purpose engineering
  test_df$Survived <- NA
  combi <- rbind(train_df, test_df) 
  
  #Features engineering
  combi$Name <- as.character(combi$Name)
  
  # The number of titles are reduced to reduce the noise in the data
  combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
  combi$Title <- sub(' ', '', combi$Title)
  #table(combi$Title)
  combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
  combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
  combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
  combi$Title <- factor(combi$Title)
  
  # Reuniting the families together
  combi$FamilySize <- combi$SibSp + combi$Parch + 1
  combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
  combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
  combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
  #table(combi$FamilyID)
  combi$FamilyID <- factor(combi$FamilyID)
  
  
  # Decision trees model to fill in the missing Age values
  Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, data=combi[!is.na(combi$Age),], method="anova")
  combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])
  
  # Fill in the Embarked and Fare missing values
  #which(combi$Embarked == '')
  combi$Embarked[c(62,830)] = "S"
  combi$Embarked <- factor(combi$Embarked)
  #which(is.na(combi$Fare))
  combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)
  
  # Creating a new familyID2 variable that reduces the factor level of falilyID so that the random forest model
  # can be used
  combi$FamilyID2 <- combi$FamilyID
  combi$FamilyID2 <- as.character(combi$FamilyID2)
  combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
  combi$FamilyID2 <- factor(combi$FamilyID2)
  
  return(combi)
}

# Splitting back to the train and test sets
data <- feature_eng(train, test)

train <- data[1:891,]
test <- data[892:1309,]

library(xgboost)
require(xgboost)
require(methods)
require(data.table)
require(magrittr)
library(caret)

str(train)
str(test)

dmy <- dummyVars("~.", data = data)
complete2 <- data.frame(predict(dmy, newdata = data))
completematrix <- as.matrix(complete2)

train <- completematrix[1:891,]
test <-completematrix[892:1309,]

#save target variable column name

y<-train$Survived

nameCol <- names(train)[2]

#Remove target variable from training set
train$Survived <- NULL

#numberOfClasses <- max(y)+1


param <- list("objective" = "binary:logistic", eval_metric="logloss")

cv.nround <- 200
cv.nfold <- 5

xgboost_cv = xgb.cv(param=param, data = train[, -c(2)], label = train[, c(2)], nfold = cv.nfold, nrounds = cv.nround, prediction = TRUE, Verbose = FALSE)
#xgboost_cv$dt
min.merror.idx = which.min(xgboost_cv$dt[, test.logloss.mean]) 
min.merror.idx 

#build Model
fit_xgboost <- xgboost(param =param, data = train[, -c(2)], label = train[, c(2)], nrounds=min.merror.idx)




# Get the feature real names
names <- dimnames(train)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = fit_xgboost)

# Plotting
xgb.plot.importance(importance_matrix)

# Prediction on test and train sets
pred_xgboost_test <- predict(fit_xgboost, test[, -c(2)])
pred_xgboost_train <- predict(fit_xgboost, train[, -c(2)])


proportion <- sapply(seq(.3,.7,.01),function(step) c(step,sum(ifelse(pred_xgboost_train<step,0,1)!=train[, c(2)])))
dim(proportion)
predict_xgboost_train <- ifelse(pred_xgboost_train<proportion[,which.min(proportion[2,])][1],0,1)
head(predict_xgboost_train)
score <- sum(train[, c(2)] == predict_xgboost_train)/nrow(train)
score


predict_xgboost_test <- ifelse(pred_xgboost_test<proportion[,which.min(proportion[2,])][1],0,1)
tested <- as.data.frame(test)
submit <- data.frame(PassengerId = data[892:1309,c("PassengerId")], Survived = predict_xgboost_test)
write.csv(submit, file = "secondxgboost.csv", row.names = FALSE)
