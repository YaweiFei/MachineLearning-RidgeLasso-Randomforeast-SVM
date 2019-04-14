# Diamonds - MSBA Team 1-6
# Stella Wang, Yawei Fei, Brandon Finch, Michael Marcy

rm(list=ls())

####################################################
###                Functions                     ###
####################################################
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}

####################################################
###          Load required packages              ###
####################################################
needed <- c('glmnet', 'e1071','rminer', 'ISLR',
            'ggplot2','car','ROCR', 'randomForest',
            'caret','kernlab','ggthemes','FNN',"NbClust",
            "factoextra")  
installIfAbsentAndLoad(needed)


####################################################
###              Data Preprocessing              ###
####################################################
set.seed(54)

#load our dataset and explore the data
diamonds <- read.table("diamonds.csv", header=T, sep=',')
dim(diamonds)
summary(diamonds)

#get rid of the rowid column
diamonds <- diamonds[2:11] 
str(diamonds)
head(diamonds)

#drop rows with zeros (20 rows) and NAs
na.omit(diamonds)
diamonds[diamonds == 0] <- NA
diamonds<-na.omit(diamonds)
origdata<-diamonds

#create model matrix and generate dummy variables
#for our categorical variables
diamonds.x <- model.matrix(price ~ ., diamonds)[, -1]

#scale our dataset
diamonds.x <- scale(diamonds.x)
head(diamonds.x)

#look at price distribution
avgprice = mean(diamonds$price)
h = hist(diamonds$price)
h$density = h$counts/sum(h$counts)*100
plot(h,freq=FALSE) #percent of differnet prices

#look at the correlation between variables
cor(cbind(diamonds.x,price = diamonds$price))

#create train and test data sets
n <- nrow(diamonds)
train.indices.credit <- sample(n, .8 * n)
train.x <- diamonds.x[train.indices.credit, ]
test.x <- diamonds.x[-train.indices.credit, ]
train.y <- diamonds$price[train.indices.credit]
test.y <- diamonds$price[-train.indices.credit]
train.data <- data.frame(train.x, price=train.y)
test.data <- data.frame(test.x, price=test.y)


# Combine scaled test and train data so we have a scaled full dataset
diamond_sc <- rbind(train.data, test.data)

####################################################
###         Model 1: Linear Regression           ###
####################################################
set.seed(54)

#linear regression taken all the variables into consideration
trainset=data.frame(train.x,train.y)
lmfit<-lm(train.y~.,data=trainset)
testset=data.frame(test.x,test.y)

#calculate the mse for train dataset
y_pred_train<-predict(lmfit,data=trainset)
(mse_lmfitall_train<-mean((y_pred_train-train.y)^2))

#calculate the mse for test dataset
y_pred<-predict(lmfit,data=testset)
(mse_lmfitall<-mean((y_pred-test.y)^2))
#overfitting for mse_train << mse_test

#linear regression exploration
lm1<-lm(origdata$price~.,data=origdata)
(VIF<-vif(lm1))
VIF[,1]

#remove from the biggest vif, so we remove x
regressdata1<-origdata[,-(8)]
lm2<-lm(regressdata1$price~.,data=regressdata1)
VIF<-vif(lm2)
VIF[,1]

# remove z
regressdata2<-regressdata1[,(-9)]
lm3<-lm(regressdata2$price~.,data=regressdata2)
VIF<-vif(lm3)
VIF[,1]

#remove y
regressdata3<-regressdata2[,(-8)]
lm4<-lm(regressdata3$price~.,data=regressdata3)
VIF<-vif(lm4)
VIF[,1]
par(mfrow=c(2,2))
plot(lm4)
summary(lm4)
#calculate the mse for test dataset
y_predlmselect<-predict(lm4,newdata=regressdata3[-train.indices.credit,])
(mselmselect<-mean((y_predlmselect-test.y)^2))

#calculate the mse for train dataset
y_predlmselect_train<-predict(lm4,newdata=regressdata3[train.indices.credit,])
(mselmselect_train<-mean((y_predlmselect_train-train.y)^2))

par(mfrow=c(1,1))


####################################################
###           Model 2: KNN Regression            ###
####################################################
set.seed(54)

krange <- seq(1, 19, by = 1)
knnmse <- rep(0,19)

for(k in krange) {
  validate.pred <- knn.reg(train.x, 
                           test.x,  
                           train.y, 
                           k = k)
  
  knnmse[k] <- mean((test.y -validate.pred$pred)^2)
}
knnmse
minknnmse<-min(knnmse)
which.min(knnmse)


####################################################
###          Model 3: Ridge Regression           ###
####################################################
set.seed(54)

#Create a λ grid vector of 100 elements ranging from 10^10 to 10^-10
grid <- 10 ^ seq(10, -10, length=100)

#create a ridge model
mod.ridge <- glmnet(train.x,train.y,alpha=0,lambda=grid)
plot(mod.ridge, xvar='lambda', label=T)

#Using the cv.glmnet() function
cv.out.class <- cv.glmnet(train.x, train.y, alpha=0)

#Plot these MSE against the log of λ
plot(cv.out.class)

#Display the best cross-validated lambda value
(bestlam <- cv.out.class$lambda.min)

#calculate training mse
ridge.pred_train <- predict(mod.ridge, s=bestlam, newx=train.x)
(mse.ridge_train <- mean((ridge.pred_train-train.y)^2))

#create a vector of test set predictions and compute the associated test error
ridge.pred <- predict(mod.ridge, s=bestlam, newx=test.x)
(mse.ridge <- mean((ridge.pred-test.y)^2))


####################################################
###          Model 4: Lasso Regression           ###
####################################################
set.seed(54)

#create a lasso model
mod.lasso <- glmnet(train.x,train.y,alpha=1,lambda=grid)
plot(mod.lasso, xvar='lambda', label=T)

#Using the cv.glmnet() function
cv.out.class <- cv.glmnet(train.x, train.y, alpha=1)

#Plot these MSE against the log of λ.
plot(cv.out.class)

#Display the best cross-validated lambda value
(bestlam <- cv.out.class$lambda.min)

#calculate training mse
lasso.pred_train <- predict(mod.lasso, s=bestlam, newx=train.x)
(mse.lasso_train <- mean((lasso.pred_train-train.y)^2))

#create a vector of test set predictions and compute the associated test error.
lasso.pred <- predict(mod.lasso, s=bestlam, newx=test.x)
(mse.lasso <- mean((lasso.pred-test.y)^2))


#Display the coefficients of the model associated with the best λ.
(lasso.coefficients <- predict(mod.lasso, type="coefficients", s=bestlam))


###################################################
####     Model 5: Random Forest Regression     ####        
###################################################
set.seed(54)

#build a random forest model with ntree=300
#time difference of 14.90014 mins
t1<-Sys.time()
rf.random<-randomForest(price~.,data=train.data,ntree = 300,importance=T)
rf.random
t2<-Sys.time()
t2-t1
plot(rf.random,main = "Trend of training Error along with the number of tree")


oob.err<-double(23)
test.err<-double(23)
t1<-Sys.time()
#mtry is no of Variables randomly chosen at each split
#this for loop will take more than 2 hours
for(mtry in 1:23) {
  rf=randomForest(price ~ . ,data = train.data,mtry=mtry,ntree=100) 
  oob.err[mtry] = rf$mse[100] #Error of all Trees fitted
  pred<-predict(rf,test.data) #Predictions on Test Set for each Tree
  test.err[mtry]= with(test.data, mean((test.y - pred)^2)) #Mean Squared Test Error
  cat(mtry," ")}
t2<-Sys.time()
t2-t1
matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))


#build our final random forest model mtry=4, ntree=100 
#Time difference of 2.21675 mins
t1<-Sys.time()
rf.model<-randomForest(price~.,data=train.data,mtry=4,ntree = 100,importance=T)
rf.model
t2<-Sys.time()
t2-t1
plot(rf.model,main = "Trend of training Error along with the number of tree")
importance(rf.model)
varImpPlot(rf.model)

#predict the price using this model
rf.predict<-predict(rf.model,test.data)
plot(test.y,c(test.y-rf.predict))

#calculate the mse for test dataset
(mse.rf<- mean((test.y-rf.predict)^2))

#mse comparison
rbind(mse_lmfitall, mse.ridge, mselmselect,mse.lasso, minknnmse, mse.rf)


##############################################
###     Model 6: Classification - SVM      ###
##############################################
set.seed(54)

# Let's look at the price points for 50%, 75%, and 90% of the data
quantile(diamonds$price, c(0.50, 0.75, 0.90))
# We want to predict diamonds in the top 10% of price
topten <- quantile(diamonds$price, 0.90)

train_class <- train.data
test_class <- test.data

# Cut at 90th percentile, above = 1 and below = 0
# Change training data price to binary variable
for (i in 1:nrow(train_class)){
  if (train_class$price[i] >= topten){
    train_class$price[i] <- 1}
  else {train_class$price[i] <- 0}
}
# Change testing data price to binary variable
for (i in 1:nrow(test_class)){
  if (test_class$price[i] >= topten){
    test_class$price[i] <- 1}
  else {test_class$price[i] <- 0}
}

table(train_class$price)
table(test_class$price)

# Change price to a factor
train_class$price <- as.factor(train_class$price)
test_class$price <- as.factor(test_class$price)

# Combine binary test and train data so we have a full dataset for use later
class_full <- rbind(train_class,test_class)


# Test different cost parameters to get best SVC
# The tuning step took ~2 hours to run
tune.out <- tune(svm,price ~ ., data=train_class, kernel="linear", 
                 ranges=list(cost=c(0.01, 0.1, 1, 5, 10, 20, 30, 40,
                                    50, 60, 70, 80, 90, 100)))

summary(tune.out)
bestsvc <- tune.out$best.model
summary(bestsvc)

# Predict the test data using the support vector classifier
classpred <- predict(bestsvc, test_class)
class_confmatrix <- table(Actual=test_class$price, Predicted=classpred) #confusion matrix
(class_confmatrix)

# Correct predictions
(class_confmatrix["0","0"] + class_confmatrix["1","1"]) / sum(class_confmatrix)
# Error rate
(class_confmatrix["0","1"] + class_confmatrix["1","0"]) / sum(class_confmatrix)
# Type I error
(class_confmatrix["0","1"]) / sum(class_confmatrix["0",])
# Type II error
(class_confmatrix["1","0"]) / sum(class_confmatrix["1",])

# Create a support vector classifer model based on the cost parameter
# determined in the tuning steps, using the full dataset for training
svm.final <- svm(price ~ ., data = class_full, kernel="linear", 
                 cost=bestsvc$cost, scale=FALSE)
finalpred <- predict(svm.final, class_full)
class_confmatrix <- (table(Actual=class_full$price, Predicted=finalpred))
class_confmatrix

# Correct predictions
(class_confmatrix["0","0"] + class_confmatrix["1","1"]) / sum(class_confmatrix)
# Error rate
(class_confmatrix["0","1"] + class_confmatrix["1","0"]) / sum(class_confmatrix)
# Type I error
(class_confmatrix["0","1"]) / sum(class_confmatrix["0",])
# Type II error
(class_confmatrix["1","0"]) / sum(class_confmatrix["1",])

# It performs a tiny bit better than the training model, but otherwise
# very similar. Not too bad!


#########################################################
####          Model 7:  K-Means CLustering           ####        
#########################################################
set.seed(54)

# Running the elbow and silhoutte methods maxed out computer's memory, 
# so we will try it with half the dataset - elbow still doesn't work.
# Split dataset it half
halfdata <- sample(n, size=n*0.5, replace=FALSE)
halfdata <- diamond_sc[halfdata,]

# Too large a dataframe for R to handle. Need a representative sample
# 100% data too large; 80% too large; 50% too large
# Elbow Method for finding optimal number of clusters
kelbow <- fviz_nbclust(test.data, kmeans, method = "wss")
kelbow
# At 20% of data, elbow suggests 2-3 clusters

# Avg Silhouette method
ksilh <- fviz_nbclust(halfdata, kmeans, method = "silhouette")
ksilh
# Looks like 2 clusters is best (50% data used)

# Let's see if varying the starting point affects the clustering
kmeanslist <- c()
for (i in seq(5,100,5)){
  diamond_kmeans <- kmeans(x= diamond_sc, centers = 2, nstart = i)
  kmeanslist <- c(kmeanslist, diamond_kmeans$tot.withinss)
}
min(kmeanslist)
unique(kmeanslist)
# They are all equally large, not much benefit in different starting points

diamond_kmeans <- kmeans(x= diamond_sc, centers = 2, nstart = 25)
table(diamond_kmeans$cluster)

fviz_cluster(diamond_kmeans, data = diamond_sc)
