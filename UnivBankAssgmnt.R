rm(list=ls(all=T))

setwd("C:/INSOFE/Assignment 1")


# Load required libraries
library(vegan)
library(dummies)
library(e1071)
library(RColorBrewer)
library(h2o)
library(randomForest)
library(MASS)
library(DMwR)
library(rpart)
library(neuralnet)
library(FNN)
library(C50)
library(ada)
library(rpart.plot)

# Read the data using csv file
bank_data = read.csv(file = "UnivBank.csv", header = TRUE)

str(bank_data)
summary(bank_data)

# Change the column names (replace dot with underscore)
names(bank_data)[10]<-paste("Personal_Loan")
names(bank_data)[11]<-paste("Securities_Account")
names(bank_data)[12]<-paste("CD_Account")

# Removing the id, zip and experience. Experience andage are highly correlated so delete exp
drop_Attr = c("ID", "ZIP.Code", "Experience")
attr = setdiff(names(bank_data), drop_Attr)
bank_data = bank_data[, attr]
rm(drop_Attr, attr)

# Convert attribute to appropriate type  
num_Attr = c( "Age", "Income","Mortgage","CCAvg")
cat_Attr = setdiff(names(bank_data), num_Attr)
bank_data = data.frame(sapply(bank_data, as.character))
cat_Data = data.frame(sapply(bank_data[,cat_Attr], as.factor))
num_Data = data.frame(sapply(bank_data[,num_Attr], as.numeric))

bank_data = cbind(num_Data, cat_Data)

# Do the summary statistics and check for missing values and outliers.
summary(bank_data)
sum(is.na(bank_data))
# No missing values . If there were missing values do imputation or knn imputation

#________________________________Standardisation________________________________________

# Split it independent variable and dependent variab for classification.
ind_Num_ClaAttr = num_Attr
ind_Cat_ClaAttr = setdiff(cat_Attr, "Personal_Loan")
rm(num_Attr)

# Standardizing the numeric data
std_Data = decostand(bank_data[,ind_Num_ClaAttr], "range") 
rm(ind_Num_ClaAttr)

# Using dummy function, convert education and family categorical attributes into numeric attributes 
edu = dummy(bank_data$Education)
family = dummy(bank_data$Family)
std_Data = cbind(std_Data, edu, family)
ind_Cat_Attr = setdiff(ind_Cat_ClaAttr, c("Education", "Family"))
rm(edu, family)

# Using as.numeric function, convert remaining categorical attributes into numeric attributes 
std_Data=cbind(std_Data, data.frame(lapply(lapply(bank_data[ ,ind_Cat_ClaAttr],as.character),as.numeric)))
std_Data = subset(std_Data, select = -c(Family, Education))
rm(ind_Cat_Attr)
ind_Attr = names(std_Data)
str(std_Data);summary(std_Data)

# Append the Target attribute 
std_Data = cbind(std_Data, Personal_Loan=bank_data[,"Personal_Loan"])
head(bank_data) #not standarised bank data 11 attributes
head(std_Data) # standardised bank_data for classification 15 attributes

# Plotting of categorical with respect tp personal Loan

barplot(table(bank_data$Personal_Loan,bank_data$Family ), col  = brewer.pal(3,"Set1"))
legend("topright", legend=levels(bank_data$Personal_Loan), title="Personal Loan",fill =brewer.pal(3,"Set1"))

barplot(table(bank_data$Personal_Loan,bank_data$Education ), col  = brewer.pal(3,"Set1"))
legend("topright", legend=levels(bank_data$Personal_Loan), title="Personal Loan",fill =brewer.pal(3,"Set1"))

barplot(table(bank_data$Personal_Loan,bank_data$Securities_Account), col  = brewer.pal(3,"Set1"))
legend("topright", legend=levels(bank_data$Personal_Loan), title="Personal Loan",fill =brewer.pal(3,"Set1"))

barplot(table(bank_data$Personal_Loan,bank_data$CD_Account ), col  = brewer.pal(3,"Set1"))
legend("topright", legend=levels(bank_data$Personal_Loan), title="Personal Loan",fill =brewer.pal(3,"Set1"))


rm(cat_Data, cat_Attr, num_Data, ind_Attr, ind_Cat_ClaAttr)


#______________________Feature engineering_______________________________________

# 1. Principle Component Analysis (PCA) -Regression

set.seed(345)

std_RegData = std_Data
std_RegData$Personal_Loan = as.character(std_RegData$Personal_Loan)
std_RegData$Personal_Loan = as.numeric(std_RegData$Personal_Loan)

pca_regdata = prcomp(std_RegData[,-2]) 
summary(pca_regdata)
pca_regcomponents = pca_regdata$rotation[,1:9] # Returning first 9 principle components
plot(pca_regdata)
print(pca_regdata)
pca_regbankdata = pca_regdata$x[,1:9] 
head(pca_regbankdata)

# Principle Component Analysis (PCA) -Classification

pca_cladata = prcomp(std_Data[,-16]) #without the target variable , Use the standardised data
summary(pca_cladata)
pca_components = pca_cladata$rotation[,1:10] # Returning first 10 principle components
plot(pca_cladata)
print(pca_cladata)
pca_clabankdata = pca_cladata$x[,1:10]


#______________________________ Autoencoders___________________________________________


# 2. generate non linear features using Auto encoders -Regression

h2o.init(ip='localhost', port=54321, max_mem_size = '1g')
set.seed(123)

# Import a local R  data frame (bank_data)to the H2O cloud
bank_data.hex <- as.h2o(x = bank_data, destination_frame = "bank_data.hex")
y = "Income"
x = setdiff(colnames(bank_data.hex), y)

aec = h2o.deeplearning(x = x, autoencoder = T,
                       training_frame = bank_data.hex,reproducible = T,
                       activation = "Tanh",
                       hidden = c(20), epochs = 100)

regfeatures = as.data.frame(h2o.deepfeatures(data = bank_data.hex[,x], 
                                          object = aec,layer = 1))

head(regfeatures) 


# 2. generate non linear features using Auto encoders -Classification

y = "Personal_loan"
x = setdiff(colnames(bank_data.hex), y)

aec_Cla = h2o.deeplearning(x = x, autoencoder = T,
                        training_frame = bank_data.hex,reproducible = TRUE,
                        activation = "Tanh",
                        hidden = c(20), epochs = 100)

clafeatures = as.data.frame(h2o.deepfeatures(data = bank_data.hex[,x], 
                                                 object = aec_Cla,layer = 1))

head(clafeatures)
h2o.shutdown(prompt = F)

#________________________Creating Regression data_________________________________

# add extracted features with original data - Regression Data
new_bank_regdata = data.frame(bank_data, pca_regbankdata,regfeatures) # cobining with original data

new_bankStd_regdata = data.frame(std_RegData, pca_regbankdata,regfeatures) #combining with standard data


#________________________Creating Classification data_________________________________

# add extracted features with original data - Classification Data
new_bank_cladata = data.frame(bank_data, pca_clabankdata,clafeatures)
head(new_bank_cladata)

new_bankStd_cladata = data.frame(std_Data, pca_clabankdata,clafeatures)
head(new_bankStd_cladata)

#________________________Random Forest(Regression)______________________________________________

# Build the regression model using randomForest

rf_bankregdata = randomForest(Income ~ ., data = new_bank_regdata, keep.forest=TRUE, ntree=30)
# importance of attributes
importanceValues = data.frame(attribute=rownames(round(importance(rf_bankregdata), 2)),
                              MeanDecreaseGini = round(importance(rf_bankregdata), 2))
importanceValues = importanceValues[order(-importanceValues$IncNodePurity),]

# plot (directly prints the important attributes) 
varImpPlot(rf_bankregdata)

# Top 10 or 11 Important attributes
TopImpRgAttrs = as.character(importanceValues$attribute[1:10])

TopImpRgAttrs

rm(rf_bankregdata)

## Build the regression model using randomForest with standardised bank data
rf_bankStdRegData = randomForest(Income ~ ., data=new_bankStd_regdata, keep.forest=TRUE, ntree=30)

# importance of attributes
importanceValues = data.frame(attribute=rownames(round(importance(rf_bankStdRegData), 2)),
                              MeanDecreaseGini = round(importance(rf_bankStdRegData), 2))
importanceValues = importanceValues[order(-importanceValues$IncNodePurity),]

# plot (directly prints the important attributes) 
varImpPlot(rf_bankStdRegData)

# Top 10 or 11 Important attributes
TopImpStdRgAttrs = as.character(importanceValues$attribute[1:10])

TopImpStdRgAttrs
rm(rf_bankStdRegData)

#________________________Random Forest(Classification)______________________________________________

set.seed(776543)
rf_bankcladata = randomForest(Personal_Loan ~ ., data=new_bank_cladata, keep.forest=TRUE, ntree=30)

# importance of attributes
round(importance(rf_bankcladata), 2)
importanceValues = data.frame(attribute=rownames(round(importance(rf_bankcladata), 2)),
                              MeanDecreaseGini = round(importance(rf_bankcladata), 2))
importanceValues = importanceValues[order(-importanceValues$MeanDecreaseGini),]

# plot (directly prints the important attributes) 
varImpPlot(rf_bankcladata)

# Top 10 Important attributes
TopImpClaAttrs = as.character(importanceValues$attribute[1:9])

TopImpClaAttrs

# random forest for standardised bank data
rf_Stdbankcladata = randomForest(Personal_Loan ~ ., data=new_bankStd_cladata, keep.forest=TRUE, ntree=30)

# importance of attributes
round(importance(rf_Stdbankcladata), 2)
importanceValues = data.frame(attribute=rownames(round(importance(rf_Stdbankcladata), 2)),
                              MeanDecreaseGini = round(importance(rf_Stdbankcladata), 2))
importanceValues = importanceValues[order(-importanceValues$MeanDecreaseGini),]

# plot (directly prints the important attributes) 
varImpPlot(rf_Stdbankcladata)

# Top 10 Important attributes
TopImpStdClaAttrs = as.character(importanceValues$attribute[1:8])

TopImpStdClaAttrs


rm(rf_Stdbankcladata, importanceValues, rf_bankcladata)


#_______________________ Split data to train and test (Regression)_____________________________________

reg_Data = cbind(new_bank_regdata[,TopImpRgAttrs],Income = new_bank_regdata[,"Income"])

set.seed(1234)
train_index = sample(x = nrow(reg_Data), size = 0.6*nrow(reg_Data))
train_RegData = reg_Data[train_index,]
test_RegData = reg_Data[-train_index,]

# For Standarised Data
stdReg_Data = cbind(new_bankStd_regdata[,TopImpStdRgAttrs],Income = new_bankStd_regdata[,"Income"])
set.seed(15343)
train_index = sample(x = nrow(stdReg_Data), size = 0.6*nrow(stdReg_Data))
train_StdRegData = stdReg_Data[train_index,]
test_StdRegData = stdReg_Data[-train_index,]
rm(train_index)

#_______________________ Split data into train and test for classification ___________________


Cla_Data = cbind(new_bank_cladata[,TopImpClaAttrs],Personal_Loan = new_bank_cladata[,"Personal_Loan"])
set.seed(1544)
train_index = sample(x = nrow(Cla_Data), size = 0.6*nrow(Cla_Data))
train_ClaData = Cla_Data[train_index,]
test_ClaData = Cla_Data[-train_index,]


Cla_StdData = cbind(new_bankStd_cladata[,TopImpStdClaAttrs],Personal_Loan = new_bankStd_cladata[,"Personal_Loan"])
set.seed(154984)
train_index = sample(x = nrow(Cla_StdData), size = 0.6*nrow(Cla_StdData))
train_StdClaData = Cla_StdData[train_index,]
test_StdClaData = Cla_StdData[-train_index,]



#______________________Linear Regression _______________________________________________________

#Building linear regression
names(train_RegData)
LinReg_BankData = lm(Income~., data=train_RegData)
summary(LinReg_BankData)

# StepAIC uing direction = "bacward". This first takes all the attributes and then drops one by one to find the model with least AIC value
LinRegAIC_1 = stepAIC(LinReg_BankData, direction = "backward")
summary(LinRegAIC_1)  # AIC=16894.46

#Error verification on train data
regr.eval(train_RegData$Income, LinRegAIC_1$fitted.values)

#Error verification on test data
linear_RegPredTest = predict(LinRegAIC_1,test_RegData)
regr.eval(test_RegData$Income, linear_RegPredTest)

# Plots to check regression assumptions
par(mfrow = c(2,2))
#plot(LinRegVIF_4)
plot(LinRegAIC_1)
par(mfrow = c(1,1))


#________________________Decision Tree (Regression)____________________________________


dtCart = rpart(Income ~.,data=train_RegData,method="anova")    
summary(dtCart)

plot(dtCart,main="Decision Tree for Income",uniform=TRUE)
text(dtCart,cex = 0.7,use.n = TRUE,xpd =TRUE)

prp(dtCart, faclen = 0, cex = 0.5, extra = 0)
prp(dtCart, faclen = 0, cex = 0.5, extra = 1)

predCartTrain = predict(dtCart, newdata=train_RegData, type="vector")
predCartTest = predict(dtCart, newdata=test_RegData, type="vector")


regr.eval(train_RegData[,"Income"], predCartTrain, train.y = train_RegData[,"Income"])
regr.eval(test_RegData[,"Income"], predCartTest, train.y = train_RegData[,"Income"])



#__________________________________SVM (Regression) ________________________________________


# Build best SVM model 
names(train_StdRegData)
svm_reg = svm(x = train_StdRegData[,1:10], 
            y = train_StdRegData[,11],
            type = "nu-regression", 
            kernel = "linear", cost = 1e-7) 

# Look at the model summary
summary(svm_reg)

# Predict on train data and check the performance
pred_trainSVN = predict(svm_reg, train_StdRegData[,1:10])
regr.eval(train_StdRegData$Income, pred_trainSVN)

# Predict on test data and check the performance 
pred_testSVN = predict(svm_reg, test_StdRegData[,1:10])
regr.eval(test_StdRegData$Income,pred_testSVN )

# Hyperparameter tuning 
tuned <- tune.svm(x = train_StdRegData[,1:10], 
                  y = train_StdRegData[,11], 
                  type = "nu-regression", 
                  gamma = 10^(-6:-1), cost = 10^(1:2))
summary(tuned)
tunedmodel = tuned$best.model
regr.eval(train_StdRegData$Income, predict(tunedmodel, train_StdRegData[,1:10]))
regr.eval(test_StdRegData$Income, predict(tunedmodel, test_StdRegData[,1:10]))
#rm(test_Data, train_Data, tuned)

#____________________________________Neural Network__________________________________________


formula = as.formula(paste("Income ~", 
                           paste(setdiff(names(train_StdRegData),"Income"), 
                                 collapse = " + ")))
nn_regModel = neuralnet(formula, data=train_StdRegData, hidden=2)


# Plot the neural network
plot(nn_regModel)

# Remove target attribute from Test Data
test_Data_No_Target = subset(test_StdRegData, select=-c(Income))

# Predict on test
nn_Testpredict = compute(nn_regModel, covariate= test_Data_No_Target)
rm(test_Data_No_Target)

# Predict on train
nn_Trainpredict = compute(nn_regModel, covariate= subset(train_StdRegData, select=-c(Income)))

# View the predicted values
nn_Testpredict$net.result


#____________________________________ KNN ______________________________________________

pred_KnnRegTrain = knn.reg(train = train_StdRegData[,1:10], 
                     test = train_StdRegData[,1:10],
                     y = train_StdRegData$Income, k = 6)

pred_KnnRegTest = knn.reg(train = train_StdRegData[,1:10], 
                    test = test_StdRegData[,1:10],
                    y = train_StdRegData$Income, k = 6)

pred_RegKnnTrain = pred_KnnRegTrain$pred
pred_RegKnnTest = pred_KnnRegTest$pred
regr.eval(train_StdRegData[,"Income"],pred_RegKnnTrain )
regr.eval(test_StdRegData[,"Income"], pred_RegKnnTest)

#rm(pred_Train, pred_Test)



#__________________________________ Random Forest (Regression)_________________________________________


rf_bankregModel = randomForest(Income ~ ., data=train_RegData, keep.forest=TRUE, ntree=30)
summary(rf_bankregModel)
# plot (directly prints the important attributes) 
varImpPlot(rf_bankregModel)

# Print and understand the model
print(rf_bankregModel)

# Important attributes
model_Imp$importance  

# Predict on Train data 
pred_RFRegTrain = predict(rf_bankregModel, train_RegData[,1:10],
                     type="response", norm.votes=TRUE)

# Predict on Test data
pred_RFRegTest = predict(rf_bankregModel, test_RegData[,1:10],
                          type="response", norm.votes=TRUE)


regr.eval(train_RegData$Income, pred_RFRegTrain)
regr.eval(test_RegData$Income, pred_RFRegTest)


#________________________ GBM (Regression)________________________________________________________________


localh2o = h2o.init(nthreads = -1)

train.h2o = as.h2o(train_RegData)
test.h2o = as.h2o(test_RegData)


gbm_RegModel = h2o.gbm(y = "Income",x = setdiff(names(train.h2o),"Income"),
                    training_frame = train.h2o,distribution= "gaussian")


h2o.performance(gbm_RegModel)
predict_gbmTrain = as.data.frame(h2o.predict(gbm_RegModel,newdata = train.h2o[,setdiff(names(train.h2o),"Income")]))
predict_gbmTest = as.data.frame(h2o.predict(gbm_RegModel,newdata = test.h2o[,setdiff(names(test.h2o),"Income")]))

regr.eval(train_RegData$Income, predict_gbmTrain)
regr.eval(test_RegData$Income, predict_gbmTest)



#___________________________________9.Deep Learning______________________________


dl_Regmodel = h2o.deeplearning(x=setdiff(names(train.h2o),"Income"),
                          y="Income",
                          seed=18834,
                          training_frame = train.h2o,
                          nfolds = 3,
                          stopping_rounds = 7,
                          epochs = 400,
                          overwrite_with_best_model = TRUE,
                          activation = "Tanh",
                          input_dropout_ratio = 0.1,
                          hidden = c(10,10),
                          l1=6e-4,
                          loss = "Automatic",
                          distribution = "AUTO",
                          stopping_metric = "RMSE")

plot(dl_Regmodel)
predict_dlTrain = as.data.frame(h2o.predict(dl_Regmodel,newdata = train.h2o[,setdiff(names(train.h2o),"Income")]))
predict_dlTest = as.data.frame(h2o.predict(dl_Regmodel,newdata = test.h2o[,setdiff(names(test.h2o),"Income")]))


regr.eval(train_RegData$Income, predict_dlTrain)
regr.eval(test_RegData$Income, predict_dlTest)



#________________________________ Stacking (Regression) _________________________________

stack_Traindata = data.frame(DL = predict_dlTrain,
                        GBM = predict_gbmTrain,
                        RandomForest = pred_RFRegTrain,
                        SVN = pred_trainSVN,
                        KNN = pred_RegKnnTrain,
                        CART = predCartTrain,
                        NN = nn_Trainpredict$net.result,
                        LinearReg = LinRegAIC_1$fitted.values, 
                        Income = train_RegData$Income)

head(pred_KnnRegTest)
names(stack_data)
stack_lm = lm(Income~. , data = stack_Traindata)
summary(stack_lm)

# Check the "ensemble_Model model" on the train data
ensemble_Train = predict(stack_lm, stack_Traindata[,1:8])
                         

#---------Predict on Test Data----------

stack_testData = data.frame(DL = predict_dlTest,
                            GBM = predict_gbmTest,
                            RandomForest = pred_RFRegTest,
                            SVN = pred_testSVN,
                            KNN = pred_RegKnnTest,
                            CART = predCartTest,
                            NN = nn_Testpredict$net.result,
                            LinearReg = linear_RegPredTest, 
                            Income = test_RegData$Income)


# Check the "glm_ensemble model" on the test data
ensemble_Test = predict(stack_lm, stack_testData[,1:8])

regr.eval(train_RegData$Income, ensemble_Train)
regr.eval(test_RegData$Income, ensemble_Test)


####################################################################################################
 #                                  CLASSIFICATION
###############################################################################################

#________________________________Logistic Regresion ______________________________________

logReg_model = glm(Personal_Loan ~ .,data=train_ClaData,
                      family="binomial")

summary(logReg_model) #AIC :20

AIC_model = stepAIC(logReg_model, direction = "backward")

summary(AIC_model)

#Predicting with the model

predlogReg_train = predict(object = AIC_model,
                                  newdata = train_ClaData[,-which(names(train_ClaData) == "Personal_Loan")] ,type="response")


threshold = 0.25 #setting threshold

predlogReg_train[predlogReg_train > threshold]=1
predlogReg_train[predlogReg_train <= threshold]=0

#Building confusion matrix and calculating the evaluation metrics
cm_log_train = table(train_ClaData[,which(names(train_ClaData) == "Personal_Loan")], predlogReg_train)
cm_log_train
LRacc = sum(diag(cm_log_train))/nrow(train_ClaData)
LRrec = cm_log_train[2,2]/sum(cm_log_train[2,])
LRpre = cm_log_train[2,2]/sum(cm_log_train[,2])


#Applying the same on test data
predlogReg_test = predict(object = AIC_model, 
                               newdata = test_ClaData[,-which(names(test_ClaData) == "Personal_Loan")],type="response")

predlogReg_test[predlogReg_test>threshold]=1
predlogReg_test[predlogReg_test<=threshold]=0
cm_log_test = table(test_ClaData[,which(names(test_ClaData) == "Personal_Loan")],
                    predlogReg_test)
LRacc_test = sum(diag(cm_log_test))/nrow(test_ClaData) #accuracy
LRrec_test = cm_log_test[2,2]/sum(cm_log_test[2,]) #recall
LRpre_test = cm_log_test[2,2]/sum(cm_log_test[,2]) #


#_____________________________Decision Trees c5.0 _________________________________________

#Decision Trees using C5.0 (For Classification Problem)
dtC50 = C5.0(Personal_Loan ~ ., data = train_ClaData, rules=TRUE)
summary(dtC50)
C5imp(dtC50, pct=TRUE)

dtc50_predTrain = predict(dtC50, newdata=train_ClaData, type="class")
cmtrain = table(train_ClaData$Personal_Loan, dtc50_predTrain)
rcTrain =(cmtrain[2,2])/(cmtrain[2,1]+cmtrain[2,2])*100
accu_Train= sum(diag(cmtrain))/sum(cmtrain)

dtc50_predTest = predict(dtC50, newdata=test_ClaData, type="class")
cmtest =table(test_ClaData$Personal_Loan,dtc50_predTest )
rcTest=(cmtest[2,2])/(cmtest[2,1]+cmtest[2,2])*100
accu_Test= sum(diag(cmtest))/sum(cmtest)

rm(a,rcEval,rcTest,rcTrain)

#____________________________Decision Trees CART____________________________________________

#Decision Trees using CART (For Classification Problem)

dtCartCla = rpart(Personal_Loan~.,data = train_ClaData, method ="class")    
plot(dtCartCla,main="Classification Tree for loan Class",margin=0.15,uniform=TRUE)
text(dtCartCla,use.n=T)
prp(dtCartCla,faclen = 1)
summary(dtCartCla)

cart_ClaPred = predict(dtCartCla, newdata=train_ClaData, type="class")
a=table(train_ClaData$Personal_Loan, cart_ClaPred)
(a[2,2])/(a[2,1]+a[2,2])*100

accu_Train= sum(diag(a))/sum(a)

cart_ClaTestPred = predict(dtCartCla, newdata=test_ClaData, type="class")
a=table(test_ClaData$Personal_Loan,cart_ClaTestPred )
accu_Test= sum(diag(a))/sum(a)



#___________________________ SVM (Classification) __________________________________________

# Build best SVM model 
snmCla_Model = svm(x = train_StdClaData[,1:8], 
            y = train_StdClaData$Personal_Loan, 
            type = "C-classification", 
            kernel = "linear", cost = 10, gamma = 0.1) 

# Look at the model summary
summary(snmCla_Model)

snmCla_Model$index

# Predict on train data  
svmpred_Train  =  predict(snmCla_Model, train_StdClaData[,1:8])  

# Build confusion matrix and find accuracy   
cm_Train = table(train_StdClaData$Personal_Loan, svmpred_Train)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
#rm(pred_Train, cm_Train)

# Predict on test data
SVMpred_Test = predict(snmCla_Model, test_StdClaData[,1:8]) 

# Build confusion matrix and find accuracy   
cm_Test = table(test_ClaData$Personal_Loan, SVMpred_Test)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
#rm(pred_Test, cm_Test)

accu_Train

#___________________________________ Neural Net (Classification)____________________________

train_StdClaData$Personal_Loan = as.character(train_StdClaData$Personal_Loan)
train_StdClaData$Personal_Loan = as.numeric(train_StdClaData$Personal_Loan)

formula = as.formula(paste("Personal_Loan ~", 
                           paste(setdiff(names(train_StdClaData),"Personal_Loan"), 
                                 collapse = " + ")))
nn_cla = neuralnet(formula, data=train_StdClaData, hidden=2)



# See covariate and result variables of neuralnet model
out_model_class = cbind(nn_cla$covariate, nn_cla$net.result[[1]])


# To view top records in the data set
head(out_model_class) 
rm(out_model_class)

# Plot the neural network
plot(nn_cla)


# Remove target attribute from Train and Test Data
train_No_Target = subset(train_StdClaData, select=-c(Personal_Loan))
test_No_Target = subset(test_StdClaData, select=-c(Personal_Loan))

# Predict on train and test
predTrain_nn = compute(nn_cla, covariate= train_No_Target)
predTest_nn = compute(nn_cla, covariate= test_No_Target)

# View the predicted values
head(predTrain_nn$net.result)
nn_cla$weights
nn_cla$result.matrix

# Compute confusion matrix and calculate recall for Train Data
predicted_nnTrain = factor(ifelse(predTrain_nn$net.result > 0.5, 1, 0))
conf_Matrix = table(train_StdClaData$Personal_Loan, predicted_nnTrain)
recall = (conf_Matrix[2,2]/(conf_Matrix[2,1]+conf_Matrix[2,2]))*100
accu_Test= sum(diag(conf_Matrix))/sum(conf_Matrix)
recall

# Compute confusion matrix and calculate recall for Train Data
predicted_nnTest = factor(ifelse(predTest_nn$net.result > 0.5, 1, 0))
conf_Matrix = table(test_StdClaData$Personal_Loan, predicted_nnTest)
recall = (conf_Matrix[2,2]/(conf_Matrix[2,1]+conf_Matrix[2,2]))*100
accu_Test= sum(diag(conf_Matrix))/sum(conf_Matrix)
recall



#_________________________________ KNN ____________________________________________________

# k = 1
pred_ClaKnnTrain = knn(train_StdClaData[,1:8], 
                 train_StdClaData[,1:8], 
                 train_StdClaData$Personal_Loan, k = 1)

cm_Train = table(pred_ClaKnnTrain, train_StdClaData$Personal_Loan)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

pred_ClaKnnTest = knn(train_StdClaData[,1:8], 
                test_StdClaData[,1:8], 
                train_StdClaData$Personal_Loan, k = 1)

cm_Test = table(pred_ClaKnnTest, test_StdClaData$Personal_Loan)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)

accu_Train
accu_Test

#_______________________________ Adaboost ________________________________________________


Ada_model = ada(x = train_StdClaData[,1:8], 
            y = train_StdClaData$Personal_Loan, 
            iter=20, loss="logistic") # 20 Iterations if this is more overfitting so better reduce the itteration

# Look at the model summary
model
summary(Ada_model)

# Predict on train data  
pred_AdaClasTrain  =  predict(Ada_model, train_StdClaData[,1:8])  

# Build confusion matrix and find accuracy   
cm_Train = table(train_StdClaData$Personal_Loan, pred_AdaClasTrain)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm( cm_Train)

# Predict on test data
pred_AdaClaTest = predict(Ada_model, test_StdClaData[,1:8]) 

# Build confusion matrix and find accuracy   
cm_Test = table(test_StdClaData$Personal_Loan, pred_AdaClaTest)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
rm( cm_Test)

accu_Train
accu_Test

#_______________________________ Random Forest ____________________________________________


rfCla_Model = randomForest(Personal_Loan ~ ., data=train_ClaData, keep.forest=TRUE, ntree=30)
summary(rfCla_Model)
# plot (directly prints the important attributes) 
varImpPlot(rfCla_Model)

# Print and understand the model
print(rfCla_Model)


# Predict on Train data 
pred_ClaRegTrain = predict(rfCla_Model, train_ClaData[,1:9],
                          type="response", norm.votes=TRUE)

# Predict on Test data
pred_ClaRegTest = predict(rfCla_Model, test_ClaData[,1:9],
                         type="response", norm.votes=TRUE)


# Build confusion matrix and find accuracy   
cm_Train = table(train_ClaData$Personal_Loan, pred_ClaRegTrain)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm( cm_Train)



# Build confusion matrix and find accuracy   
cm_Test = table(test_ClaData$Personal_Loan, pred_ClaRegTest)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
rm( cm_Test)

accu_Train
accu_Test

#__________________________________ GBM _________________________________________________



localh2o = h2o.init(nthreads = -1)

train.h2o = as.h2o(train_ClaData)
test.h2o = as.h2o(test_ClaData)

names(train.h2o)

gbm_ClaModel <- h2o.gbm(model_id = "GBM.hex", ntrees = 100, 
               learn_rate=0.01, max_depth = 4,  distribution = "bernoulli", 
               y = "Personal_Loan", x = setdiff(names(train.h2o), "Personal_Loan"),
               training_frame = train.h2o)


h2o.performance(gbm_ClaModel)

predict_gbmClaTrain = h2o.predict(gbm_ClaModel,newdata = train.h2o[,setdiff(names(train.h2o), "Personal_Loan")])
predict_gbmClaTest = h2o.predict(gbm_ClaModel,newdata = test.h2o[,setdiff(names(test.h2o), "Personal_Loan")])

dataTrain_GBM = h2o.cbind(train.h2o[,"Personal_Loan"], predict_gbmClaTrain)
dataTest_GBM = h2o.cbind(test.h2o[,"Personal_Loan"], predict_gbmClaTest)

# Copy predictions from H2O to R
pred_ClaGBM_Tain = as.data.frame(dataTrain_GBM)
pred_ClaGBM_Test = as.data.frame(dataTest_GBM)


 
# Build confusion matrix and find accuracy   
conf_Matrix_GBM = table(pred_ClaGBM_Tain$Personal_Loan, pred_ClaGBM_Tain$predict)
accu_Train= sum(diag(conf_Matrix_GBM))/sum(conf_Matrix_GBM)

conf_Matrix_GBM = table(pred_ClaGBM_Test$Personal_Loan, pred_ClaGBM_Test$predict)
accu_Train= sum(diag(conf_Matrix_GBM))/sum(conf_Matrix_GBM)





#___________________________________9.Deep Learning______________________________


dl_Clamodel = h2o.deeplearning(x=setdiff(names(train.h2o),"Personal_Loan"),
                               y="Personal_Loan",training_frame=train.h2o,
                               seed=183234,
                               activation = "RectifierWithDropout",
                               hidden = c(20,20),
                               input_dropout_ratio = 0.2,
                               l1 = 1e-5,
                               epochs = 10)
plot(dl_Clamodel)
predict_dlClaTrain = h2o.predict(dl_Clamodel,newdata = train.h2o[,setdiff(names(train.h2o),"Personal_Loan")])
predict_dlClaTest = h2o.predict(dl_Clamodel,newdata = test.h2o[,setdiff(names(test.h2o),"Personal_Loan")])
head(predict_dlClaTest)

dataTrain_dl = h2o.cbind(train.h2o[,"Personal_Loan"], predict_dlClaTrain)
dataTest_dl = h2o.cbind(test.h2o[,"Personal_Loan"], predict_dlClaTest)

# Copy predictions from H2O to R
pred_Cladl_Tain = as.data.frame(dataTrain_dl)
pred_Cladl_Test = as.data.frame(dataTest_dl)

conf_Matrix_dl = table(pred_Cladl_Tain$Personal_Loan, pred_Cladl_Tain$predict)
accu_Train= sum(diag(conf_Matrix_dl))/sum(conf_Matrix_dl)

conf_Matrix_dl = table(pred_Cladl_Test$Personal_Loan, pred_Cladl_Test$predict)
accu_Train= sum(diag(conf_Matrix_dl))/sum(conf_Matrix_dl)



#________________________________ Stacking (Regression) _________________________________

stackCla_Traindata = data.frame(DL = pred_Cladl_Tain$predict,
                             GBM = pred_ClaGBM_Tain$predict,
                             RandomForest = pred_ClaRegTrain,
                             Adaboost = pred_AdaClasTrain,
                             SVN = svmpred_Train,
                             KNN = pred_ClaKnnTrain,
                             CART = cart_ClaPred,
                             C50 = dtc50_predTrain,
                             NN = predicted_nnTrain,
                             #LogisticReg = predlogReg_train, 
                             Personal_Loan = train_ClaData$Personal_Loan)

#head(pred_KnnRegTest)
str(stackCla_Traindata)
test_Pred_All_Models = data.frame(sapply(stackCla_Traindata, as.factor))
test_Pred_All_Models$LogisticReg = as.character(test_Pred_All_Models$LogisticReg)
stack_glm = glm(Personal_Loan~. , data = test_Pred_All_Models)
summary(stack_glm)

# Check the "ensemble_Model model" on the train data
ensemble_Train = predict(stack_glm, stackCla_Traindata[,1:8])


#---------Predict on Test Data----------

stack_ClatestData = data.frame(DL = pred_Cladl_Test$predict,
                            GBM = pred_ClaGBM_Test$predict,
                            RandomForest = pred_ClaRegTest,
                            Adaboost = pred_AdaClaTest,
                            SVN = SVMpred_Test,
                            KNN = pred_ClaKnnTest,
                            CART = cart_ClaTestPred,
                            C50 = dtc50_predTest,
                            NN = predicted_nnTest,
                            LogisticReg = predlogReg_test, 
                            Personal_Loan = test_ClaData$Personal_Loan)

# Check the "glm_ensemble model" on the test data
ensemble_Test = predict(stack_glm, stack_ClatestData[,1:8])

conf_Matrix = table(train_ClaData$Personal_Loan, ensemble_Train)
accu_Train= sum(diag(conf_Matrix))/sum(conf_Matrix)
rm( cm_Train)

conf_Matrix_1 = table(test_ClaData$Personal_Loan, pred_Cladl_Test$ensemble_Test)
accu_Train= sum(diag(conf_Matrix_1))/sum(conf_Matrix_1)



