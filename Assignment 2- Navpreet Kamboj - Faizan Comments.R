# Applied Machine Learning for Health Data - University of Toronto
# Assignment 2
# Author: Navpreet Kamboj
# Date: Nov 16, 2020

getwd()

#import data
#data <- read.csv('/Users/navpreetkamboj/Documents/projects/ML course/assignment/cleveland.txt',header=F, na.strings="?")	

data = read.csv('cleveland.txt',header=F, na.strings="?")
dim(data)
summary(data)
  
# Creating a binary outcome -- Value 0: < 50% diameter narrowing Value 1: > 50% diameter narrowing

data$V14<- factor(ifelse(data$V14 >=1, 1, 0),levels = c(0,1))
prop.table(table(data$V14))
summary(data$V14)
library(Hmisc)
describe(data)

#dummy variables
data$V2<- factor(data$V2)
data$V3 <- factor(data$V3)
data$V6 <- factor(data$V6)
data$V7 <- factor(data$V7)
data$V9 <- factor(data$V9)
data$V11 <- factor(data$V11)
data$V13 <- factor(data$V13)
summary(data)

#2- predictors 

predictors <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13")
summary(predictors)

mydata <- data[,c("V14", predictors)]
dim(mydata)
summary(mydata)

#removing missing values- there are 6 missing values.
mydata.nonmis<- na.omit(mydata)
dim(mydata.nonmis)
summary(mydata.nonmis)
table(mydata.nonmis$V14)
plot(mydata.nonmis$V14, main= "Distrubution of Heart Disease", xlab= "Heart Disease (0 = Absence 1= Present)")

library(Hmisc)
describe(mydata.nonmis)


# Create outer fold/loops

k_outer_fold <- 5

set.seed(123)
folds <- rep_len(1:k_outer_fold, nrow(mydata.nonmis))
folds <- sample(folds, nrow(mydata.nonmis))

brier.cv.logreg <- c()
auc.cv.logreg <- c()

#actual outer loop (rep 5 times)
for(i in 1:k_outer_fold){

  #i = 1
  fold <- which(folds == i)
  length(fold)
  
  # Creating the outter fold for the outer loop which is rep 5 times.
  training.I <- sample(nrow(mydata.nonmis[fold,]),round(nrow(mydata.nonmis[fold,])*.8)) 
  
  
  training <- mydata.nonmis[training.I,]
  test_final_20per <- mydata.nonmis[-training.I,]
  
  all.data <- model.matrix(V14 ~.,data=training)[,-1] #this gives us a matrix of the predictors 
  summary(all.data)
  # we need to standardize the data, in order to make things comparable
  all.data <- scale(all.data)
  all.data
  
  # Debudding checks
  length(training.I)
  length(training)
  length(test_final_20per)
  #

  # Second split for teain and validation sets for tunning. 
  #train model and cross validation
  set.seed(123)
  train.I <- sample(nrow(training),round(nrow(training)*2/3)) 
  
  train <- training[train.I,]
  val <- training[-train.I,]
  
  
  #Classification 1- Logistic regression with regularization- lASSO
  library(glmnet)
  
  x_train <- model.matrix(V14 ~.,train)[,-1]
  y_train <-train$V14
  
  x_val <- model.matrix(V14 ~.,val)[,-1]
  y_val<-val$V14
  
  #lasso
  
  ls.mod <- glmnet(x_train,y_train,family="binomial",alpha=1) #x is the matrix of the predictor #alpha=1 lasso #since y is a continous variable family="gaussian" is not necessary 
  plot(ls.mod, label = T)
  summary(ls.mod)
  
  set.seed(123)
  cv.ls.auc <- cv.glmnet(x_train,y_train,alpha=1,family = "binomial", type.measure = "auc")
  plot(cv.ls.auc)
  cv.ls.auc$lambda.min # 0.01114
  cv.ls.auc #91%
  
  coef.min.ls <- coef(cv.ls.auc, s = "lambda.min")
  coef.min.ls
  
  #removed V1 and V7-1 and V13-6
  
  cv.ls.mse <- cv.glmnet(x_train,y_train,alpha=1,family = "binomial", type.measure = "mse")
  plot(cv.ls.mse)
  cv.ls.mse$lambda.min #0.002290341
  cv.ls.mse
  coef.min <- coef(cv.ls.mse, s = "lambda.min")
  coef.min
  
  #misclassification error
  cv.ls.cl <- cv.glmnet(x_train,y_train,alpha=1,family = "binomial", type.measure = "class")
  plot(cv.ls.cl)
  cv.ls.cl$lambda.min #0.004392668
  cv.ls.cl
  coef.min <- coef(cv.ls.cl, s = "lambda.min")
  coef.min
  
  #classification with KNN
  library(class)
  library(epiR)
  library(pROC)
  
  
  apply(all.data,2,mean)
  apply(all.data,2,sd)
  
  # all.data = training_scaled # scaled will not have labels.
  
  train <- all.data[train.I,]
  val <- all.data[-train.I,]
  
  labels.train <- training$V14[train.I]  #this gives us the classes
  labels.val <- training$V14[-train.I]
  
  #predictions <- knn(train,val,labels.train,k=3)   
  #t <- table(predictions, labels.val)[2:1,2:1] #you want the positive to be first
  #epi.vals(t)  #positive predictive value- 1/10 the number of val outcome that are + out of total. 
  
  set.seed(123)
  library(caret)
  i=1
  k.optm=1
  
  for(i in 1:100){
    knn.mod2 <- knn(train=train, test=val, cl=labels.train, k=i)
    k.optm[i] <- 100 * sum(labels.test == knn.mod2)/NROW(labels.val)
    k=i
    cat(k,'=',k.optm[i],'
    ')
  } 
  #KNN of 14 is optimal 
  
  # using k = 14
  
  predictions14 <- knn(train,val, cl =labels.train,k=14, prob = T)
  t14 <- table(predictions14, labels.val)[2:1,2:1]
  epi.vals(t14) 
  summary(predictions14)
  my.prob <- attr(predictions14,"prob")
  my.prob
  predictions14[1] #winning class is 1 #MY PROF WANTED US TO USE THIS BUT I'M NOT TOO SURE WHAT TO DO- HE SAID WE CAN USE IT AND THEN DO A LOOP USING IFELSE TO GET AUC
  my.prob.correct[1] < 1 - my.prob[1]
  
  #ifelse(predictions14, 1, my.prob) 
  
  tab <- table(predictions14, labels.val)
  tab
  accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
  accuracy(tab) #what is this providing? is this for ROC?
  #83% accuracy 
  
  #CV
  set.seed(123)
  predictions <- knn.cv(all.data, training$V14, k=100, prob = T) #by adding .cv we are doing the cross validation. If we don;t privide a seperate data set for classification, it does it automatically 
  t2 <- table(predictions, training$V14)[2:1,2:1]
  epi.vals(t2)
  my.prob.cv <- attr(predictions,"prob")
  my.prob.cv
  predictions[1] #1 is winning class
  
  tab2 <-table(predictions, training$V14)
  accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
  accuracy(tab2)
  
  #79% accuracy for training set
  
  #classification with classification trees
  library(tree)
  library(epiR)
  library(pROC)
  
  V14_fac <- factor(train$V14, levels = 0:1, labels = c("No", "Yes")) #converting to factor (yes no)
  
  cfit <- tree(V14 ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13,
               data = train) #I used the dataset with missing values would that be ok?
  summary(cfit)
  plot(cfit)
  text(cfit, pretty = 0)
  
  preds0 <- predict(cfit,type="class")
  preds0 
  length(preds0)
  summary(preds0)
  
  #includes dataset with the missing values
  preds <- predict(cfit,newdata=train,type="class") 
  summary(preds)
  length(preds)
  
  t <- table(preds, V14)[2:1,2:1]
  epi.vals(t)
  sum(diag(t))/sum(t)
  table(preds, V14)
  
  # let's look at the predicted probabilities
  pred.probs <- predict(cfit,newdata=train,type="vector")
  head(pred.probs) 
  pred.probs.heart <- pred.probs[,2]
  
  roc1 <- roc(V14~pred.probs.heart) 
  plot(roc1)
  roc1
  #96.5% very high because it's training dataset
  
  #train data and valing it 
  set.seed(123)
  length(train.I) #SAMPLE SIZE IS 198
  tmp.tree <- tree(V14 ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13,
                   data = train)
  pred.probs <- predict(tmp.tree,newdata=val,type="vector") #new data is original - training set
  pred.probs.ht <- pred.probs[,2]
  roc1 <- roc(V14[-train.I]~pred.probs.ht)
  plot(roc1)
  auc <- roc1$auc
  auc
  
  # about 77% for val
  
  # Finding opt pruned level for pruned tree.
  set.seed(322)
  cv.res <- cv.tree(cfit, FUN=prune.tree, method = "misclass", K = 5)  
  cv.res
  
  # Prune the tree using opt k = 8 found in cv above.
  pruned <- prune.misclass(cfit,best=8)
  plot(pruned)
  text(pruned,pretty=0)
  
  tree.pred <- predict(pruned,newdata=val,type="vector")
  pred.probs.heart <- tree.pred[,2]
  roc2 <- roc(V14[-train.I____val___]~pred.probs.heart)
  plot(roc2)
  auc2 <- roc2$auc
  auc2
  
  #Area under the curve for pruned tree is 0.87
  
  #auc.cv.logreg[i] <- myroc$auc
  # Store all the results in a data frame.
  #results_df
  
  ################### Inner fold ends #########################
  
  
  
  # Test data set test_final_20per ## For outter loop.
  
  # Fit best model
  # train data and valing it 

  tmp.tree <- tree(V14 ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13,
                   data = training)
  
  pred.probs <- predict(tmp.tree,newdata=test_final_20per,type="vector") #new data is original - training set
  pred.probs.ht <- pred.probs[,2]
  roc1 <- roc(V14[-train.I____test_______]~pred.probs.ht)
  plot(roc1)
  auc <- roc1$auc
  auc
  
  # about 77% for val
  
  #prune tree
  set.seed(322)
  cv.res <- cv.tree(cfit, FUN=prune.tree, method = "misclass", K = 5) 
  cv.res
  
  pruned <- prune.misclass(cfit,best=8)
  plot(pruned)
  text(pruned,pretty=0)
  
  tree.pred <- predict(pruned,newdata=test_final_20per,type="vector")
  pred.probs.heart <- tree.pred[,2]
  roc2 <- roc(V14[-train.I]~pred.probs.heart)
  plot(roc2)
  auc2 <- roc2$auc
  auc2
  
  
  # Store results
  # results_df
  
}






