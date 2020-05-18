library(dplyr)
library(ggplot2)
library(corrplot)
library(data.table)

# Importing the data
df = read.csv("F:/Mayur/ANALYTICS/Data scientist/Data scienc with R/Diabetes prediction model R shiny/data/diabetes.csv")

# Reviewing the data
names(df)

##==============================================Data set information==========================##
# Pregnancies = Number of pregnancies had happpened
# Glucose = Glucose level present in the body
# Bloodpressure, Skinthickness, Insulin = Variables which explains the health of the person
# BMI = Body mass index (to understand whether the person is underweight, fit or overweight
# DiabetesPedigreeFunction = Probability of having diabetes
# Age = Age of the person
# Outcome = Dependent variable 0 = No diabetes, 1 = Diabetes Yes
##============================================================================================##

head(df)
dim(df) # 768 rows and 9 columns
colSums(is.na(df)) # no missing data in the data set
summary(df)

# minimum age of the person is 21
# Bloodpressure, BMI, Glucose = 0 is not correct
# finding out the records with 0 blood pressure

View(df %>% filter(BloodPressure == 0)) 
# 35 variables with 0 blood pressure
# around 4% of total data hence removing the records with 0 BP

df_1 = df %>% filter(BloodPressure>0)
df = NULL

str(df_1)
# converting outcome variable to factor
df_1$Outcome = as.factor(df_1$Outcome)
View(df_1)

# checking summary again
summary(df_1)
# checking records with 0 BMI
df_1 %>% filter(BMI==0)
df_1 %>% filter(Glucose==0)
View(df_1 %>% filter(SkinThickness==0))
View(df_1 %>% filter(Insulin==0))

df_1 %>% filter(Insulin==0) %>% group_by(Outcome) %>% summarise(Count = n())
# insulin 0 is too low insulin, if person has insulin below 3 
# then the person could have type 1 diabetes
# from the table we can see that non diabetic people are high in numbers

# reviewing records with 0 insulin and diabteic
View(df_1 %>% filter(Insulin==0,Outcome==1))
# Data captured under Insulin and Skinthickness variable is doubtful

# removing records with glucose = 0 because it is low blood sugar case 
# the data has 3 non diabetic people which seems doubtful
# also removing records with BMI = 0
# As we are using this data for shiny dashboard demonstration
# we can keep Insulin and Skinthickness though it has incorrect data
# removing these two variables will impact on predictions


df_2 = df_1 %>% filter(BMI > 0) %>% 
                filter(Glucose > 0) 
summary(df_2)

# 17 pregnancies also seem to be unusual
# as our data is small lets consider it

# checking corelation
c = cor(df_2[,-9])
View(as.data.frame(c))
round(c,2)
corrplot(c,method = "color")

# age has high corelation with pregnancies
# bloodpressure has some corelation with age
# considerable corelation between Insulin and Skinthickness
# Also, Skinthickness has considerable corelation with BMI


# changing column name of outcome to diabetic
names(df_2)[names(df_2)=="Outcome"] = "Diabetic"
# 0 is no diabetes and 1 is diabetes

# understanding relationship dependent variable
ggplot(df_2,aes(x = Diabetic, y = DiabetesPedigreeFunction))+geom_boxplot()+
        ggtitle("Pedigree Function and diabetes")


ped_fun = data.table(df_2 %>% group_by(Diabetic) %>% summarise(min_pedigree = round(min(DiabetesPedigreeFunction),2),
                                         mean_pedigree = round(mean(DiabetesPedigreeFunction),2),
                                         median_pedigree = round(median(DiabetesPedigreeFunction),2),
                                         max_pedigree = round(max(DiabetesPedigreeFunction),2)
                                         ))
ped_fun

# total diabetic and non diabetic in the data set
table(df_2$Diabetic)
prop.table(table(df_2$Diabetic))

# relation between pregnancies and diabetes
ggplot(df_2,aes(x= Pregnancies,fill = Diabetic))+geom_bar(position = "fill")+
        ggtitle("Pregnancies vs Diabetes")+ylab("Percentage Share")
# high percentage share of diabetic with pregnancies >=7

# reviewing records with prgnancies >=14
df_2 %>% filter(Pregnancies>=14)

# checking relationship between BMI and bloodpressure
ggplot(df_2,aes(x=BloodPressure,y=BMI,shape=Diabetic,color=Diabetic))+
    geom_point()+
    geom_smooth(method=lm, se=FALSE, fullrange=TRUE)+
    ggtitle("Bloodpressure vs BMI")    


# developing prediction model
library(caret)
# shuffling the data
set.seed(100)
model_data = sample(nrow(df_2)) # shuffling the rows
model_df = df_2[model_data,] # creating data with shuffled rows

# splitting data into train and test
split_rows = createDataPartition(model_df$Diabetic,p=0.7,list = FALSE)

Train = model_df[split_rows,]
Test = model_df[-split_rows,]


# developing logistic regression model
log_model = train(Diabetic~.,data = Train,method = "glm",family="binomial")
summary(log_model)

pred = predict(log_model,Test)
confusionMatrix(pred,Test$Diabetic)

# checking area under curve using ROCR library
library(ROCR)
ROCRpred <- prediction(as.integer(pred),as.integer(Test$Diabetic))
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE)
auc <- performance(ROCRpred, measure = "auc")
auc <- auc@y.values[[1]]
auc
# 76 %

# developing decision tree model
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
# trctrl = trainControl(method = "repeatedcv",number = 10,repeats = 3)
# set.seed(101)
# dtree_fit = train(Diabetic ~., data = Train, method = "rpart",
#                     parms = list(split = "gini"),
#                     trControl=trctrl,
#                    tuneLength=10)
# dtree_fit

dtree_fit = rpart(Diabetic~.,data = Train,method = "class",
                  control = rpart.control(minsplit = 40,xval = 10))

dtree_fit
rpart.plot(dtree_fit)
dtree_fit$cptable

best_cp = dtree_fit$cptable[which.min(dtree_fit$cptable[,"xerror"]),"CP"]

dtree_fit = rpart(Diabetic~.,data = Train,method = "class",
                  control = rpart.control(minsplit = 30,cp=best_cp,xval = 10))

dtree_fit
rpart.plot(dtree_fit)
rpart.rules(dtree_fit)

pred = predict(dtree_fit,Test,type = "class")
confusionMatrix(pred,Test$Diabetic)


# predictions using SVM
library(e1071)

# scaling the data set
train_scale = scale(Train[-9])
test_scale = scale(Test[-9])

train_scale = cbind(train_scale,Train[9])
test_scale = cbind(test_scale,Test[9])

svm_model = svm(Diabetic~.,data = train_scale,type = "C-classification",kernel = "radial")
svm_model
summary(svm_model)

pred = predict(svm_model,test_scale,type = "class")
confusionMatrix(pred,test_scale$Diabetic)

# tuning SVM model
tuned_svm = tune.svm(Diabetic~.,data = train_scale,
                     gamma = c(0.0001,0.001,0.015,0.01,0.1,0.3,0.5,1.0,10,12,15),
                     cost = c(0.0001,0.001,0.015,0.01,0.1,0.3,0.5,1.0))

tuned_svm

svm_model = svm(Diabetic~.,data = train_scale,type = "C-classification",kernel = "radial",
                gamma=tuned_svm$best.parameters$gamma,cost=tuned_svm$best.parameters$cost)


pred = predict(svm_model,test_scale,type = "class")
confusionMatrix(pred,test_scale$Diabetic)


# Checking prediction score using Random forest
library(randomForest)

set.seed(102)
rf_model = randomForest(Diabetic~.,data = Train)
rf_model

# tuning RF model
rf_tune = tuneRF(x = subset(Train,select = -Diabetic),
                 y = Train$Diabetic,
                 ntreeTry = 500)

rf_tune
                 

rf_model = randomForest(Diabetic~.,data = Train,
                        mtry = rf_tune[,"mtry"][which.min(rf_tune[,"OOBError"])],
                        ntree=500,
                        importance = TRUE)

rf_model

pred = predict(rf_model,Test,type = "class")
confusionMatrix(pred,Test$Diabetic)

# checking AUC
ROCRpred = prediction(as.integer(pred),as.integer(Test$Diabetic))
ROCRperf = performance(ROCRpred,"tpr","fpr")
plot(ROCRperf,colorize=TRUE)                      

auc = performance(ROCRpred,measure = "auc")
auc@y.values[[1]]
#73.36%

# logistics regression model is predicting the diabetes better than other three models

# saving logisitcs model
saveRDS(log_model,"Diabetes_Pred_model.RDS")

