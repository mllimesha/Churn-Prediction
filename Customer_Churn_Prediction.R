library(ggplot2)
library(dplyr)
library(glmnet)
library(pROC)
library(car)
library(MASS)
library(rpart)
library(ROCR)
library(caret)
library(class)
library(e1071)
library(randomForest)
library(mlbench)
library(lift)

telco = read.csv("Telco-Customer-Churn.csv", header = TRUE)
str(telco)

take_telco <- telco[, -1]
take_telco$SeniorCitizen <- as.factor(take_telco$SeniorCitizen)
levels(take_telco$Churn) <- c(0,1)
take_telco[rowSums(is.na(take_telco)) > 0,]

take_telco <- na.omit(take_telco)

##Random Forest##
rf_data_telco <- na.omit(take_telco)
rf_data_telco$Churn = as.character(rf_data_telco$Churn)
rf_data_telco$churn = factor(rf_data_telco$Churn) 


set.seed(1)
n_rf = nrow(rf_data_telco)
rf_trainIndex = sample(1:n_rf, size = round(0.8*n_rf), replace=FALSE)
rf_train_telco = rf_data_telco[rf_trainIndex ,]
rf_test_telco = rf_data_telco[-rf_trainIndex ,]

rf_telco <- randomForest(formula = as.factor(Churn)~ ., data = rf_data_telco, importance = TRUE, ntree = 200)
telco_var_importance <- data.frame(importance(rf_telco))
varImpPlot(rf_telco)

telco_crossval <- rfcv(rf_data_telco[,c(1:19)], as.factor(rf_data_telco$Churn), cv.fold = 10, step = .5)
with(telco_crossval, plot(n.var, error.cv, log="x", type="o", lwd=2, xlab="Number of Variables", ylab="Error Rate"))

rf_final_telco <- randomForest(formula = as.factor(Churn) ~ ., data = rf_train_telco, importance = TRUE, ntree = 500)
rf_final_telco

##Calculate the accuracy on training data.##

rf_train_telco <- rf_train_telco %>% mutate(Imp_preds = predict(rf_final_telco, rf_train_telco))
rf_train_accuracy <- mean(rf_train_telco$Churn==rf_train_telco$Imp_preds)
rf_train_accuracy

##Calculate the accuracy on test data.##

rf_test_telco <- rf_test_telco %>% mutate(Imp_preds = predict(rf_final_telco, rf_test_telco))
rf_testaccuracy <- mean(rf_test_telco$Churn==rf_test_telco$Imp_preds)
rf_testaccuracy


##SVM##

svm_data_telco <- na.omit(take_telco)
set.seed(1)
n_svm = nrow(svm_data_telco)
svm_trainIndex = sample(1:n_svm, size = round(0.8*n_svm), replace=FALSE)
svm_telco_train = svm_data_telco[svm_trainIndex ,]
svm_telco_test = svm_data_telco[-svm_trainIndex ,]

svm_fit <- svm(as.factor(Churn) ~ ., probability = TRUE, data = svm_telco_train)
svm_fit

svm_preds_train <- predict(svm_fit, svm_telco_train, probability = TRUE)
svm_train_probs <- data.frame(attr(svm_preds_train, "probabilities"))
svmrocplot_train_simple <- plot(roc (svm_telco_train$Churn, svm_train_probs$Yes))

svmroc_train_simple <- roc(svm_telco_train$Churn, svm_train_probs$Yes)
svmauc_train_simple <- auc(svmroc_train_simple)
svmauc_train_simple

svm_preds_test <- predict(svm_fit, svm_telco_test, probability = TRUE)
svm_test_probs <- data.frame(attr(svm_preds_test, "probabilities"))
svmrocplot_test <- plot(roc(svm_telco_test$Churn, svm_test_probs$Yes))

svmroc_test <- roc(svm_telco_test$Churn, svm_test_probs$Yes)
svmauc_test <- auc(svmroc_test)
svmauc_test


##Logistic Regression##

lr_good_telco <- na.omit(take_telco)
levels(lr_good_telco$StreamingMovies)[levels(lr_good_telco$StreamingMovies)=="No internet service"] <- "No"
levels(lr_good_telco$OnlineSecurity)[levels(lr_good_telco$OnlineSecurity)=="No internet service"] <- "No"
levels(lr_good_telco$OnlineBackup)[levels(lr_good_telco$OnlineBackup)=="No internet service"] <- "No"
levels(lr_good_telco$DeviceProtection)[levels(lr_good_telco$DeviceProtection)=="No internet service"] <- "No"
levels(lr_good_telco$TechSupport)[levels(lr_good_telco$TechSupport)=="No internet service"] <- "No"
levels(lr_good_telco$StreamingTV)[levels(lr_good_telco$StreamingTV)=="No internet service"] <- "No"

lr_data_telco <- na.omit(lr_good_telco)
set.seed(1)
n_lr = nrow(lr_data_telco)
lr_trainIndex = sample(1:n_lr, size = round(0.8*n_lr), replace=FALSE)
lr_telco_train = lr_data_telco[lr_trainIndex ,]
lr_telco_test = lr_data_telco[-lr_trainIndex ,]

logr_telco <- glm(as.factor(Churn) ~., family = binomial, data = lr_telco_train)
logr_aic_telco <- stepAIC(logr_telco, direction = "backward")

logr_telco_train <- glm(as.factor(Churn) ~ OnlineSecurity + Dependents + TechSupport + SeniorCitizen + MonthlyCharges + StreamingTV + StreamingMovies + TotalCharges + PaymentMethod + MultipleLines + PaperlessBilling + InternetService + Contract + tenure, family = binomial, data = lr_telco_train)
summary(logr_telco_train)

P_train = predict(logr_telco_train, newdata = lr_telco_train, type = "response")
lr_train_roc_plot <- plot(roc(lr_telco_train$Churn, P_train))
auc(lr_train_roc_plot)

P_test = predict(logr_telco_train, newdata = lr_telco_test, type = "response")
lr_test_roc_plot <- plot(roc(lr_telco_test$Churn, P_test))

auc(lr_test_roc_plot)


test_train_predict_telco <- lr_telco_train %>% mutate(P_TEST = predict(logr_telco_train, newdata = lr_telco_train, type = "response"),
                                                      P_RESP = ifelse(P_TEST >= 0.26, 1, 0))
mean(test_train_predict_telco$P_RESP == lr_telco_train$Churn)

lr_telco_test <- lr_telco_test %>% mutate(P_TEST = predict(logr_telco_train, newdata = lr_telco_test, type = "response"),
                                          P_RESP = ifelse(P_TEST >= 0.26, 1, 0))
true_churn <- lr_telco_test$Churn
predicted_churn <- lr_telco_test$P_RESP
lr_accuracy <- mean(true_churn==predicted_churn)
lr_accuracy

##Confusion Matrix##

mat1 <- as.matrix(table(lr_telco_test$P_RESP, lr_telco_test$Churn))
mat1

tp <- mat1[2,2]
tn <- mat1[1,1]
fp <- mat1[1,2]
fn <- mat1[2,1]

accuracy <- (tp+tn)/(tp+fp+tn+fn)
precision <- tp/(tp+fp)

sensitivity <- tp / (tp+fn)

specificity = tn / (tn+fp)

recall <- tp/(tp+fn)
F1 <- precision*recall*2/(precision+recall)

npv <- tn/(tn+fn)

accuracy
precision
sensitivity
specificity
F1


lr_telco_test <- lr_telco_test[order(lr_telco_test$P_RESP, decreasing=TRUE),]
lr_telco_test$cumden <- cumsum(lr_telco_test$P_TEST)/sum(lr_telco_test$P_TEST)
lr_telco_test$perpop <- (seq(nrow(lr_telco_test))/nrow(lr_telco_test))*100
plot(lr_telco_test$perpop,lr_telco_test$cumden,type="l",xlab="% of Population",ylab="% of Churn")
abline(0,0.01)
