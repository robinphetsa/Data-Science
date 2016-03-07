library(caret)
library(randomForest)
library(party)
library(xgboost)
data <- read.csv("boosting.csv")

str(data)

set.seed(1234)
inTrain <- createDataPartition(data$Mpg, p=.7, list = FALSE)

train<-data[inTrain,]
test <- data[-inTrain,]


#RF_fit <- randomForest(Mpg~., data = train, ntree = 50)

#RF is taking too much resources, try gradient boosting

dmy <- dummyVars("~.", data = data)
non_normal <- data.frame(predict(dmy, newdata = data))

set.seed(1234)
inTrain <- createDataPartition(non_normal$Mpg, p=.7, list = FALSE)

train<-non_normal[inTrain,]
test <- non_normal[-inTrain,]

train.matrix <- as.matrix(train)
test.matrix <- as.matrix(test)

#train.matrix[,c(54)]
#Cross Validate
cv.nround <-200
cv.nfold <-5

param <- list("objective" = "reg:linear", "eval_metric"="rmse")

xgboost_cv <- xgb.cv(param=param, data= train.matrix[,-c(54)],
                     label = train.matrix[,c(54)],
                     nfold=cv.nfold,
                     nrounds =  cv.nround,
                     prediction = TRUE,
                     Verbose = FALSE)

min.rmse.idx = which.min(xgboost_cv$dt[, test.rmse.mean]) 
min.rmse.idx 

#Build Model
fit_xgboost_cv <- xgboost(param=param, data= train.matrix[,-c(54)],
                     label = train.matrix[,c(54)],
                     nfold=cv.nfold,
                     nrounds =  min.rmse.idx,
                     prediction = TRUE,
                     Verbose = FALSE)
#Predict
pred <-predict(fit_xgboost_cv, test.matrix[,-c(54)])
