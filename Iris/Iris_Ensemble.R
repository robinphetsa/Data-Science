#Classify Iris Data Set using Decision Trees

#First Method: Use Random Forest Ensemble 
require(party)
require(caret)
require(xgboost)
require(randomForest)

iris <- data.frame(iris)
data(iris)

#Split data set 60/40
set.seed(3456)
trainIndex <- createDataPartition(iris$Species, p = .6,
                                  list = FALSE,
                                  times = 1)

head(trainIndex)

irisTrain <- iris[ trainIndex,]
irisTest  <- iris[-trainIndex,]


RF <- randomForest(Species~., data=irisTrain, ntrees = 2000)

pred <-predict(RF, irisTest[-c(5)])

pred

confusionMatrix(pred,irisTest$Species)
#prediction accuracy of .9667
plot(margin(pred,irisTest$Species))

# 2 records of false positives

#try Conditional Inference Trees

#Build Model

CF <- cforest(Species~., data = irisTrain, controls = cforest_unbiased(mtry = 2, 
                                                                       ntree = 2000)
              )

Prediction <- predict(CF, irisTest, OOB=TRUE, type = "response")

confusionMatrix(Prediction, irisTest$Species)
#Accuracy is .98 slightly higher than RForest


#Let's try gradient boosting using the XGboost package

library(xgboost)

#Convert factors to num
irisTrain$Species <- as.numeric(irisTrain$Species)
irisTrain$Species <- as.numeric(irisTrain$Species-1)
irisTest$Species <- as.numeric(irisTest$Species)
irisTest$Species <- as.numeric(irisTest$Species-1)

train.matrix <- as.matrix(irisTrain)
test.matrix <- as.matrix(irisTest)


#Cross Validation

numberOfClasses <- max(train.matrix[]) + 1

cv.nround <- 200
cv.nfold <-2

param <- list("objective" = "multi:softprob", "eval_metric"="mlogloss", "num_class"=numberOfClasses)
xgboost_cv = xgb.cv(param=param, data = train.matrix[, -c(5)], label = train.matrix[, c(5)], nfold = cv.nfold, nrounds = cv.nround, prediction = TRUE, Verbose = FALSE)

min.mlogloss.idx = which.min(xgboost_cv$dt[, test.mlogloss.mean]) 
min.mlogloss.idx 

#Fit Gradient Boosting Model
fit_xgboost <- xgboost(param =param, data = train.matrix[, -c(5)], label = train.matrix[, c(5)], nrounds=min.mlogloss.idx)

#Predict
pred_xgboost <- predict(fit_xgboost, test.matrix[, -c(5)])

pred = matrix(pred_xgboost, nrow=numberOfClasses, ncol=length(pred_xgboost)/numberOfClasses)
pred = t(pred)
pred = max.col(pred, "last")
pred = as.numeric(pred-1)

confusionMatrix(pred,irisTest$Species)

#Accuracy .9667 same as RForest