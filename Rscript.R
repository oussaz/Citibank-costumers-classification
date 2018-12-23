library(MASS)
library(class)
library(e1071)
library(ggplot2)
library(tree)
library(randomForest)
library(Rtsne)
library(magrittr)
library(ggpubr)
library(GGally)
library(lattice)  # SMOTE
library(DMwR)     # SMOTE
library(caret)
library(scales)
library(heplots)
library(gridExtra)   
library(MVN)  # Mardia test for normality (LDA & QDA)
library(rpart) # Decision tree
library(rpart.plot)
library(rattle)  # Fancy decision tree plot
library(nnet)  # for Multinomial logistic classification
library(kableExtra)
library(gbm)  # boosting

multiplot <- function(..., plotlist = NULL, file, cols = 1, layout = NULL) {
  require(grid)
  
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  if (is.null(layout)) {
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots == 1) {
    print(plots[[1]])
    
  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    for (i in 1:numPlots) {
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


ggplotConfusionMatrix <- function(m){
  mytitle <- paste("Accuracy", percent_format()(m$overall[1]),
                   "Kappa", percent_format()(m$overall[2]))
  data_c <-  mutate(group_by(as.data.frame(m$table), Reference ), percentage = 
                      percent(Freq/sum(Freq)))
  p <-
    ggplot(data = data_c,
           aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = log(Freq)), colour = "white") +
    scale_fill_gradient(low = "white", high = "cadetblue4") +
    geom_text(aes(x = Reference, y = Prediction, label = percentage)) +
    theme(legend.position = "none") +
    ggtitle(mytitle)
  return(p)
}


credit<-read.csv('data/Credit.csv')
credit$X<-NULL

# Oversampling Asian and African American
credit1<-SMOTE(Ethnicity~.,credit)

features <- c("Income","Limit","Rating","Cards","Age","Education","Balance")
Test<-read.table('data/Test.txt',header=T)
names(credit)
summary(credit)

# Cheking for NA values
MissPercentage = function (x) {100 * sum (is.na(x)) / length (x) }
apply(credit1,2,MissPercentage)

# We will need standardized data f
credit2<-credit1
for (feat in features){
  credit2[feat][,1]<-(credit1[feat][,1]-mean(credit1[feat][,1]))/sd(credit1[feat][,1])
}
  


# Running T-SNE to visualize Data in 2D
tsne_out <- Rtsne(credit,pca=T,perplexity=30,theta=0.0) # Run TSNE
par(mfrow=c(1,1))
plot(tsne_out$Y,col=topo.colors(3), asp=1,xlab="TSN1",ylab="TSN2",border=F,main="T-SNE Transformation")
legend(10,40,legend=levels(credit$Ethnicity),col=credit$Ethnicity,fill=topo.colors(3), cex=0.7)


# Boxplots for every explanatory variable
myboxplots <- list()
i=1
for (colu in c("Income","Limit","Rating","Cards","Age","Education","Balance")){
  plt<-NULL
  plt<-ggplot(credit2,aes_string(x=colu,y=colu)) + 
  geom_boxplot(data=credit2, fill = "red", alpha = 0.2,bins = 10)
  print(plt)
  myboxplots[[i]]<-plt
  i <- i+1
}
multiplot(plotlist=myboxplots,cols=4,rows=2)

# Plotting comparaison of features' histogram comparaison of classes
myplots <- list()
i=1
for (colu in c("Income","Limit","Rating","Cards","Age","Education","Balance")){
  plt<-NULL
  plt<-ggplot(credit,aes_string(x=colu)) + 
    geom_histogram(data=subset(credit2,Ethnicity=='Caucasian'), fill = "red", alpha = 0.2,bins = 10) + 
    geom_histogram(data=subset(credit2,Ethnicity=='African American'), fill = "blue", alpha = 0.2,bins = 10) +  
    geom_histogram(data=subset(credit2,Ethnicity=='Asian'), fill = "green", alpha = 0.2, bins = 10) +  scale_fill_manual(name="group",values=c("red","blue","green"),labels=c("Cauca.","A. Ameri.","Asian"))
  print(plt)
  myplots[[i]]<-plt
  i <- i+1
}
multiplot(plotlist=myplots,cols=2,rows=4)

########################
# LDA & QDA
#######################

# LDA & QDA assumptions
# Normality
mvntest = mvn(credit2[features],mvnTest = "mardia")
# Common variance across all classes:
plotboxvar<-list()
for(i in features) {
  plotboxvar[[i]] <- ggplot(credit2, aes_string(x = "Ethnicity", y = i, col = "Ethnicity", fill = "Ethnicity")) + 
    geom_boxplot(alpha = 0.2,lwd=0.3) + 
    theme(legend.position = "none") + 
    scale_color_manual(values = c("purple", "cadetblue4","coral2")) 
  scale_fill_manual(values = c("purple", "cadetblue4","coral2"))
}

do.call(grid.arrange, c(plotboxvar, nrow = 2,ncol=4))

# Boxm test
boxm<-boxM(credit2[features],credit2$Ethnicity)
leveneTest(Cards~Ethnicity,credit2)
# Linear descriminant Analysis

lda.fit = lda(Ethnicity~Income+Limit+Rating+Cards+Age+Education+Balance,data=train)
lda.class = predict(lda.fit,test)$class
table(lda.class,test$Ethnicity)

#Randomly shuffle the data
credit2<-credit2[sample(nrow(credit2)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(credit2)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation

QDA_predval <- vector()
LDA_predval <- vector()
LDAQDA_trueval<-vector()

for(i in 1:10){
  
  #Segementing data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  # Creating test and train sets
  test <- credit2[testIndexes, ]
  train <- credit2[-testIndexes, ]
  
  # Fitting the model to the training set
  lda.fit = lda(Ethnicity~Income+Limit+Rating+Cards+Age+Education+Balance,data=train)
  qda.fit = qda(Ethnicity~Income+Limit+Rating+Cards+Age+Education+Balance,data=train)
  
  # Predicting using the test set
  lda.pred <- predict(lda.fit,test)
  qda.pred <- predict(qda.fit,test)
  
  # Concatenating the results of prediction and true values for all cross validation iterations
  LDA_predval<- c(LDA_predval,lda.pred$class)
  QDA_predval <-c(QDA_predval,qda.pred$class)
  LDAQDA_trueval<-c(LDAQDA_trueval,test$Ethnicity)
}

# Converting values to factors 
LDAQDA_facactual<-as.factor(LDAQDA_trueval)
LDA_facpred <- as.factor(LDA_predval)
QDA_facpred <- as.factor(QDA_predval)

# Confusing matrix for the CV
LDA_cfm <- confusionMatrix(LDAQDA_facactual,LDA_facpred)
LDA_cfm <- ggplotConfusionMatrix(LDA_cfm)

QDA_cfm <- confusionMatrix(LDAQDA_facactual,QDA_facpred)
QDA_cfm <- ggplotConfusionMatrix(QDA_cfm)
QDALDA_cfm <- multiplot(LDA_cfm,QDA_cfm,cols=2)
# Quadratic Discriminant Analysis

qda.fit = qda(Ethnicity~Income+Limit+Rating+Cards+Age+Education+Balance,data=train)
qda.class = predict(qda.fit,test)$class
table(qda.class,test$Ethnicity)


##################################
# SVM
##################################

# Grid search
tobj <- tune.svm(credit2[,c(1,2,3,4,5,6,11)],credit2[,10], cost= (0.1:15), gamma=0.5*(1:10),tune.control(sampling = "boot"))
summary(tobj)

# contour plot of grid search
plot(tobj, xlab = "gamma", ylab="C",nlevels = 50)

# Apparent error 0.153 for gamma = 2 cost = 6.1

#Randomly shuffle the data
set.seed(1)
credit2<-credit2[sample(nrow(credit2)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(credit2)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation

SVM_predval<-vector()
SVM_trueval<-vector()

for(i in 1:10){
  
  #Segementing data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  # Creating test and train sets
  test <- credit2[testIndexes, ]
  train <- credit2[-testIndexes, ]
  
  # Fitting the model to the training set
  svm.fit <- svm(train[,c(1,2,3,4,5,6,11)],train[,10],cost = 6.1, gamma = 2)
  
  # Predicting using the test set
  svm.pred <- predict(svm.fit,test[,c(1,2,3,4,5,6,11)])
  
  # Concatenating the results of prediction and true values for all cross validation iterations
  SVM_predval<- c(SVM_predval,as.factor(as.vector(svm.pred)))
  SVM_trueval<-c(SVM_trueval,test$Ethnicity)
}

SVM_facactual<-as.factor(SVM_trueval)
SVM_facpred <- as.factor(SVM_predval)

# Confusing matrix for the CV
SVM_cfm <- confusionMatrix(SVM_facactual,SVM_facpred)
ggplotConfusionMatrix(SVM_cfm)


###################################
# Decision tree
###################################

# Creating Training and Testing datasets
set.seed(200)
sample <- sample.int(n = nrow(credit2), size = floor(.75*nrow(credit2)), replace = F)
train <- credit2[sample, ]
test  <- credit2[-sample, ]

DT<-rpart(Ethnicity~Income+Age+Balance+Cards+Limit+Education+Rating,data=train,method="class")
summary(DT)
plot(DT)
plotcp(DT)
printcp(DT)
test$pred <- predict(DT, test, type = "class")
base_accuracy <- mean(test$Ethnicity == test$pred)

# Optimal Cp = 0.29

# Prepruning
DT_preprun <- rpart(Ethnicity~Income+Age+Balance+Cards+Limit+Education+Rating, data = train, method = "class", control = rpart.control(cp = 0, maxdepth = 8,minsplit = 100))

# Compute the accuracy of the pruned tree
test$pred <- predict(DT_preprun, test, type = "class")
accuracy_preprun <- mean(test$pred == test$Ethnicity)
rpart.plot(DT_preprun,box.palette = "RdBu",shadow.col="gray",nn=T,fallen.leaves= F,varlen = 5)

#Postpruning

# Prune the hr_base_model based on the optimal cp value
DT_postprun <- prune(DT, cp = 0.0205 )

# Compute the accuracy of the pruned tree
test$pred <- predict(DT_postprun, test, type = "class")
accuracy_postprun <- mean(test$pred == test$Ethnicity)
rpart.plot(DT_postprun,box.palette = "RdBu",shadow.col="gray",nn=T,fallen.leaves= F,varlen = 5)

# Comparaison
data.frame(base_accuracy,accuracy_preprun, accuracy_postprun)

###################################
# KNN
###################################
# Creating Training and Testing datasets
set.seed(100)
sample <- sample.int(n = nrow(credit2), size = floor(.75*nrow(credit2)), replace = F)
train <- credit2[sample, ]
test  <- credit2[-sample, ]

trControl <- trainControl(method  = "cv", number  = 10)
knn.fit <- train(Ethnicity ~ Income+Balance+Limit+Cards+Education+Age+Rating,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:10),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = train)
knn_accuracyplot<-ggplot(data = knn.fit$results,aes(k,Accuracy))+geom_point(colour="cadetblue",alpha=0.8)+geom_line(colour="cadetblue",alpha=0.6) + scale_fill_brewer(palette="Set1")
knn_trainacc <-knn.fit$results[1,]$Accuracy*100
knn.pred <- predict(knn.fit,test)
knn_cfm <- confusionMatrix(test$Ethnicity,knn.pred)
knn_testacc <- as.vector(knn_cfm$overall)[1]*100
ggplotConfusionMatrix(knn_cfm)

###################################
# Logistic Discrimination
###################################
set.seed(100)
sample <- sample.int(n = nrow(credit2), size = floor(.75*nrow(credit2)), replace = F)
train <- credit2[sample, ]
test  <- credit2[-sample, ]
multilogmodel.fit <-multinom(Ethnicity~Income+Balance+Limit+Cards+Education+Age+Rating,data=train)
multilogmodel.pred<-predict(multilogmodel.fit,as.data.frame(test),type="class")
multilog_cfm <- confusionMatrix(test$Ethnicity,multilogmodel.pred)
multilog_testacc <- as.vector(multilog_cfm$overall)[1]*100
ggplotConfusionMatrix(multilog_cfm)


# Using SVM to predict data from Test.txt

TestSVM<-read.table("data/Test.txt",header = T)
# Normalizing Test using the first data means and standard deviations
for (feat in features){
  TestSVM[feat][,1]<-(TestSVM[feat][,1]-mean(credit1[feat][,1]))/sd(credit1[feat][,1])
}

Ethnicity<-predict(svm.fit,as.data.frame(TestSVM[features]))
TestSVM<-as.data.frame(read.table("data/Test.txt",header = T))

TestSVM$Ethnicity <- Ethnicity

################################
# Random Forest
###############################
rf.fit <- randomForest(Ethnicity~Income+Balance+Limit+Cards+Education+Age+Rating,data=train,mtry=2,importance=T)
rf.pred <- predict(rf.fit,newdata = test)
rf_cfm <- confusionMatrix(test$Ethnicity,as.factor(as.vector(rf.pred)))
ggplotConfusionMatrix(rf_cfm)

################################
# Gradient Boosting
################################

set.seed(200)
sample <- sample.int(n = nrow(credit2), size = floor(.75*nrow(credit2)), replace = F)
train <- credit2[sample, ]
test  <- credit2[-sample, ]

boost.fit <- gbm(Ethnicity~Income+Balance+Limit+Cards+Education+Age+Rating,data=train,distribution="multinomial",n.trees=5000,interaction.depth = 6)
boost.pred <- predict(boost.fit,newdata=test,n.trees=5000,type="response")

p.boost.pred  <- apply(boost.pred , 1, which.max)

boost_cfm<-confusionMatrix(as.factor(as.integer(test$Ethnicity)),as.factor(p.boost.pred))
ggplotConfusionMatrix(boost_cfm)

## Using all predictors ##

###############################
# Naive Bayes
###############################

## Performing 10-fold cross validation
# Randomly shuffle the data
set.seed(1)
credit2<-credit2[sample(nrow(credit2)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(credit2)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation

NB_predval<-vector()
NB_trueval<-vector()

for(i in 1:10){
  
  #Segementing data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  # Creating test and train sets
  test <- credit2[testIndexes, ]
  train <- credit2[-testIndexes, ]
  
  # Fitting the model to the training set
  NB.fit<-naiveBayes(Ethnicity~.,data=train)
  
  # Predicting using the test set
  NB.pred <- predict(NB.fit,test)
  
  # Concatenating the results of prediction and true values for all cross validation iterations
  NB_predval<- c(NB_predval,as.factor(as.vector(NB.pred)))
  NB_trueval<-c(NB_trueval,test$Ethnicity)
}

NB_facactual<-as.factor(NB_trueval)
NB_facpred <- as.factor(NB_predval)

# Confusing matrix for the CV
NB_cfm <- confusionMatrix(NB_facactual,NB_facpred)
ggplotConfusionMatrix(NB_cfm)



#############################
# Decision tree bis
#############################
set.seed(102)
sample <- sample.int(n = nrow(credit2), size = floor(.75*nrow(credit2)), replace = F)
train <- credit2[sample, ]
test  <- credit2[-sample, ]
DT2<-rpart(Ethnicity~.,data=train,method="class",cp = 0.014)
summary(DT2)
plot(DT2)
plotcp(DT2)
printcp(DT2)
test$pred <- predict(DT2, test, type = "class")
DT2base_accuracy <- mean(test$Ethnicity == test$pred)

# Prune the hr_base_model based on the optimal cp value
DT2_postprun <- prune(DT2, cp = 0.014 )

# Compute the accuracy of the pruned tree
test$pred <- predict(DT2_postprun, test, type = "class")
DT2accuracy_postprun <- mean(test$pred == test$Ethnicity)
rpart.plot(DT2_postprun,box.palette = "RdBu",shadow.col="gray",nn=T,fallen.leaves= F,varlen = 5)
DT2_cfm<-confusionMatrix(test$Ethnicity,test$pred)
ggplotConfusionMatrix(DT2_cfm)


# Using DT2 to predict Text.txt Ethnicity

TestDT2<-read.table("data/Test.txt",header = T)
# Normalizing Test using the first data means and standard deviations
for (feat in features){
  TestDT2[feat][,1]<-(TestDT2[feat][,1]-mean(credit1[feat][,1]))/sd(credit1[feat][,1])
}

Ethnicity<-predict(svm.fit,as.data.frame(TestDT2[features]))
TestDT2<-as.data.frame(read.table("data/Test.txt",header = T))

TestDT2$Ethnicity <- Ethnicity


# SVM bis
credit3<-credit2
credit3$Male<-as.vector(as.integer(credit3$Gender)-1)
credit3$Female<-as.vector(as.integer(!credit3$Male))
credit3$Married<-as.vector(as.integer(credit3$Married)-1)
credit3$NotMarried<-as.vector(as.integer(!credit3$Married))
credit3$Student<-as.vector(as.integer(credit3$Student)-1)
credit3$NotStudent<-as.vector(as.integer(!credit3$Student))
credit3$Gender<-NULL
tobjbis <- tune.svm(credit3[,-9],credit3[,9], cost= (0.1:15), gamma=0.5*(1:10),tune.control(sampling = "boot"))


