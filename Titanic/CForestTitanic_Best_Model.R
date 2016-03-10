# Titanic using CForest

library(xgboost)
library(rpart)
library(party)
library(caret)

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



data <- feature_eng(train, test)

train <- data[1:891,]
test <- data[892:1309,]

#Using ONLY the training set so we can Cross Validate 
#Note: use both for kaggle competition

inTrain <- createDataPartition(y=train$Survived,
                               p = .75,
                               list = FALSE
)

training <- train[ inTrain,]
testing <- train[-inTrain,]

set.seed(415)
fit_CV <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction <- predict(fit_CV, testing[,-c(2)], OOB=TRUE, type = "response")

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "cforest1.csv", row.names = FALSE)

confusionMatrix(Prediction,testing$Survived)

#Correlation Coefficient of .842 or 84% accuracy on test set as opposed to 82% using Gradient Boosting
