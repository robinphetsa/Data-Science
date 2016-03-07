
#Read in CSV

data<-read.csv("HabermanSurvival.csv")
str(data)
summary(data$Age)

library(ggplot2)
ggplot(data=data, aes(data$Age))+geom_histogram()
#Bin Ages together and convert to factor

AgeGrip <- as.character(cut(data$Age, breaks = c(29,40,50,60,70,79,Inf),labels=c('30-39','40-49','50-59','60-69','70-79','80+')))
AgeGrip

data$Age <-AgeGrip
data$Age <- as.factor(data$Age)

str(data)

#Convert Surival to factor
data$Survival <- as.factor(data$Survival)

#Modeling

#Correlation Matrix
library(psych)
pairs.panels(data)

#Split into training and testing
library(caret)
Train <- createDataPartition(y=data$Survival, p=.7, list=FALSE)
training <- data[Train,]
testing <- data[-Train,]
dim(training)
dim(testing)

#Tree
library(rpart)

fit <- rpart(Survival~NumNodes+OpYear,method="class", data=training)

printcp(fit)

plotcp(fit)

post(fit)

###
library(C50)

model <- C5.0(training[-4], training$Survival)
summary(model)

#Prediction
predicted <- predict(model, testing)
table(predicted)

confusionMatrix(predicted,testing$Survival)

#Create Random Forest Model
library(randomForest)
fitRF <- randomForest(Survival~., data=training)


predicted2 <- predict(fitRF, testing)
table(predicted2)

confusionMatrix(predicted2,testing$Survival)

#NaiveBayes
library(e1071)
modFit <- naiveBayes(training[-4],training$Survival)
modFit

predicted3 <- predict(modFit, testing)
table(predicted3)

confusionMatrix(predicted3,testing$Survival)

#RF Model Two
library(randomForest)

model2 <-randomForest(Survival~., data=training)
model2

importance(model2)
predicted4 <-predict(model2, testing)
table(predicted4)

confusionMatrix(predicted4,testing$Surival)


##all models have large type II error, most likely due to unbalanced classes
