### Applied ###
rm(list = ls())
options(scipen = 999) # Convert scientific notation to numerical
#install.packages("glmnet")
#install.packages("hdm")

# Packages
library(wooldridge)
library(glmnet)
library(hdm)

# Using this dataframe
df = wage2[complete.cases(wage2),]

fml_expand = formula(wage~poly(cbind(hours,IQ, KWW, educ, exper,
                                     tenure,sibs, brthord,meduc,feduc), 
                               degree = 3,raw=TRUE)
                     +married
                     +black
                     +south
                     +urban
                     )

X = model.matrix(fml_expand, data = df)
X = X[, colnames(X)!= "(Intercept)"] 
Y = df$wage

trsize = 500 # training sample size

set.seed(2457829) # use this seed
train = sample(1:nrow(X),trsize)  
test = (-train) 
# user-defined grid
grid = 10^seq(10, -2, length = 100)

# Fit ridge on training set
ridge.mod <- glmnet(X[train,], Y[train], alpha = 0, lambda = grid, thresh = 1e-12)
# Fit LASSO on training set
lasso_mod <- glmnet(X[train,], Y[train], alpha = 1, lambda = grid)


## 1)
set.seed(2457829)
# Fitting Ridge regression model using 10-fold CV with user-defined grid
cv10fold_ug = cv.glmnet(X[train,], Y[train], alpha = 0, lambda=grid)
plot(cv10fold_ug$lambda,cv10fold_ug$cvm, main="10-fold CV user-defined grid", xlab="Lambda", ylab="CV MSE")
ridge_minlambda = cv10fold_ug$lambda.min

# From the plot, We can see that lowest CV MSE occurs at the lower values of lambda  
# let's zoom into the range that contains the minlambda value.
newgrid_2_lowerbound = 0
newgrid_2_upperbound = cv10fold_ug$lambda[50]
newgrid_2 = seq(newgrid_2_lowerbound, newgrid_2_upperbound, length.out = 100)
set.seed(2457829)
cv10fold_ug2 = cv.glmnet(X[train,], Y[train], alpha = 0, lambda=newgrid_2)
plot(cv10fold_ug2$lambda,cv10fold_ug2$cvm, main="10-fold CV user-defined grid", xlab="Lambda", ylab="CV MSE",ylim = c(100000,200000)) 
ridge_minlambda_2 = cv10fold_ug2$lambda.min

# From the plot, We can see that minimum CV MSE occurs at minlambda_2
# Therefore, let's zoom into the range between minlambda_2 to find a better minlambda
newgrid_3_lowerbound = cv10fold_ug2$lambda[which.min(cv10fold_ug2$cvm) + 1]
newgrid_3_upperbound = cv10fold_ug2$lambda[which.min(cv10fold_ug2$cvm) - 2]
newgrid_3 = seq(newgrid_3_lowerbound, newgrid_3_upperbound, length.out = 1000)
set.seed(2457829)
cv10fold_ug3 = cv.glmnet(X[train,], Y[train], alpha = 0, lambda=newgrid_3)
plot(cv10fold_ug3$lambda,cv10fold_ug3$cvm, main="10-fold CV user-defined grid", xlab="Lambda", ylab="CV MSE",ylim = c(129000,130500))
ridge_minlambda_3 = cv10fold_ug3$lambda.min

# let's settle down and call this value the best lambda
ridge_bestlambda = ridge_minlambda_3

# Evaluating its performance
ridge.pred <- predict(ridge.mod, s = ridge_bestlambda, newx = X[test,])
ridge_cv10fold_testmse = mean((ridge.pred-Y[test])^2)

# Lambda.1se performance
ridge_minlambda_1se = cv10fold_ug3$lambda.1se
ridge.pred_1se <- predict(ridge.mod, s = ridge_minlambda_1se, newx = X[test,])
ridge_cv10fold_1se_testmse = mean((ridge.pred_1se-Y[test])^2)

## Even though the CV MSE of 'lambda.min' is smaller than the CV MSE of 'lambda.1se', 
## ‘lambda.1se’ performs better than 'lambda.min' in this case as its test MSE is smaller.
## Therefore, I am inclined to believe that the 2 models does not seem to be statistically different in 
## performance on cross-validation and the 1 SE CV MSE rule of thumb seems like a good idea.


## 2)
set.seed(2457829)
# Fitting LASSO regression model using 10-fold CV with user-defined grid
cv10fold_lasso_ug= cv.glmnet(X[train,], Y[train], alpha = 1, lambda=grid)
plot(cv10fold_lasso_ug$lambda,cv10fold_lasso_ug$cvm, main="10-fold CV - LASSO - User's Grid", xlab="Lambda", ylab="CV MSE")
lasso_minlambda = cv10fold_lasso_ug$lambda.min

# From the plot, We can see that lowest CV MSE occurs at the lower values of lambda  
# let's zoom into the range that contains the lasso_minlambda value.
lasso_newgrid_2_lowerbound = 0
lasso_newgrid_2_upperbound = cv10fold_lasso_ug$lambda[60]
lasso_newgrid_2 = seq(lasso_newgrid_2_lowerbound, lasso_newgrid_2_upperbound, length.out = 100)
set.seed(2457829)
cv10fold_lasso_ug2 = cv.glmnet(X[train,], Y[train], alpha = 1, lambda=lasso_newgrid_2)
plot(cv10fold_lasso_ug2$lambda,cv10fold_lasso_ug2$cvm, main="10-fold CV - LASSO - User's Grid", xlab="Lambda", ylab="CV MSE",ylim = c(100000,200000)) 
lasso_minlambda_2 = cv10fold_lasso_ug2$lambda.min

# From the plot, We can see that minimum CV MSE occurs at lasso_minlambda_2
# Therefore, let's zoom into the range between lasso_minlambda_2 to find a better minlambda
lasso_newgrid_3_lowerbound = cv10fold_lasso_ug2$lambda[which.min(cv10fold_lasso_ug2$cvm) + 1]
lasso_newgrid_3_upperbound = cv10fold_lasso_ug2$lambda[which.min(cv10fold_lasso_ug2$cvm) - 1]
lasso_newgrid_3 = seq(lasso_newgrid_3_lowerbound, lasso_newgrid_3_upperbound, length.out = 1000)
set.seed(2457829)
cv10fold_lasso_ug3 = cv.glmnet(X[train,], Y[train], alpha = 1, lambda=lasso_newgrid_3)
plot(cv10fold_lasso_ug3$lambda,cv10fold_ug3$cvm, main="10-fold CV - LASSO - User's Grid", xlab="Lambda", ylab="CV MSE",ylim = c(129000,131000))
lasso_minlambda_3 = cv10fold_lasso_ug3$lambda.min

# let's settle down and call this value the best lambda
lasso_bestlambda = lasso_minlambda_3

# Evaluating its performance
lasso_pred <- predict(lasso_mod, s = lasso_bestlambda, newx = X[test,])
lasso_cv10fold_testmse = mean((lasso_pred-Y[test])^2)

# Lambda.1se performance
lasso_minlambda_1se = cv10fold_lasso_ug3$lambda.1se
lasso_pred_1se <- predict(lasso_mod, s = lasso_minlambda_1se, newx = X[test,])
lasso_cv10fold_1se_testmse = mean((lasso_pred_1se-Y[test])^2)

## The ‘lambda.min’ performs better than 'lambda.1se' in this case as its test MSE is smaller.
## This is also expected as it is suppose to perform better since it has the lowest CV MSE.

## 3)
# Fitting LASSO regression model with plug-in lambda
rlasso_fit = rlasso(Y[train]~X[train,],  post=FALSE)

# Evaluating performance
rlasso_pred <- predict(rlasso_fit, newdata=X[test,]) 
rlasso_testmse = mean((rlasso_pred - Y[test])^2)

# Post LASSO Estimation
rlasso_fit_post = rlasso(Y[train]~X[train,],  post=TRUE)

# Evaluating performance
postrlasso_pred  = predict(rlasso_fit_post, newdata=X[test,])
postrlasso_testmse = mean((postrlasso_pred - Y[test])^2)

## LASSO regression model with plug-in lambda has a lower test MSE as compared to the 
## Post LASSO estimation.

## 4)
Kfold=10 #number of folds
set.seed(2457829)

one_ten = rep(1:10, trsize%/%10+1) 
#note trsize%/%10+1, we +1 because trsize might not be divisible by 10
ran_ind = sample(1:trsize, trsize, replace = FALSE) 
group_cv = one_ten[ran_ind]

split = runif(trsize) #uniform random number for assigning folds
group_cv2 = as.numeric(cut(split,quantile(split, probs = seq(0,1,.1)), include.lowest = TRUE)) #groups for 10-fold cv

# Look at lambdas used in lasso above
cv_lambda = cv10fold_lasso_ug$lambda 
#extract lambdas used for our 10-fold CV previously
nlambda = length(cv_lambda) #count how many lambdas we have

Xtrainmat=X[train,]
Xtestmat =X[test,]
Ytrain   =Y[train]
Ytest    =Y[test]

cv_mses = matrix(0,Kfold,nlambda)
for(fold in 1:Kfold) {
  fold_test = group_cv == fold
  fold_train  = group_cv != fold
  #   fold_test = group_cv == fold #TRUE for those belong to test fold (only 1 fold used as test)
  #   fold_train = group_cv != fold #TRUE for those belong to train folds (9 of them)
  lasso_fold = glmnet(x= Xtrainmat[fold_train,], y = Ytrain[fold_train], lambda = cv_lambda)
  
  for(i in 1:nlambda) {
    chosencoef = predict(lasso_fold, s = cv_lambda[i], type = "coef")
    #extract coefficients
    coef_pos = which(chosencoef != 0)[-1] -1 
    #get position of non-zero coefficients, 
    ## remember to -1 as we dont want intercept, thus all indices of 
    ### the coefficients are shifted up
    
    traindf_ols = data.frame(response = Ytrain[fold_train], X = Xtrainmat[fold_train,coef_pos])
    #dataframe of included features (CVtraining only)
    
    ols_fold = lm(response ~., data = traindf_ols) 
    #run post-LASSO OLS with the retained predictors 
    testdf_ols = data.frame(response = Ytrain[fold_test], X = Xtrainmat[fold_test,coef_pos]) 
    #dataframe of included features (CVtest only)
    predicts_ols = predict(ols_fold, newdata = testdf_ols)
    cv_mses[fold,i] = mean((Ytrain[fold_test] - predicts_ols)^2) 
    #NOTE: See that the OLS is within the CV itself. 
  }
}

cvpostlasso_minlambda =cv_lambda[which.min(colMeans(cv_mses))] 
# extract the lambda minimizing CV

# Evaluating performance of CV Post LASSO 
#Step 1
train_postlasso = glmnet(X[train,],Y[train], lambda = cv_lambda)

#Step 2
bestmod_postlass = predict(train_postlasso, s = cvpostlasso_minlambda, type = "coef")

#Step 3
coefpos_chosen = which(bestmod_postlass !=0)[-1] -1
print(coefpos_chosen)

df_ols = data.frame(response = Ytrain, X = Xtrainmat[,coefpos_chosen]) 
#dataframe of  chosen features (training only)

#Step 4
ols_postlasso = lm(response ~., data = df_ols) #run post-LASSO OLS

df_test = data.frame(response = Ytest, X = Xtestmat[, coefpos_chosen]) 
#dataframe of chosen features (test only)

cvpostlasso_predict = predict(ols_postlasso, newdata = df_test)
cvpostlasso_testmse = mean((Ytest - cvpostlasso_predict)^2) #post-LASSO MSE

## The post-LASSO estimation with CV did perform better than the post-LASSO estimation
## using the plug-in Lambda as it has a lower test MSE.

## 5)
ridge.pred_OLS <- predict(ridge.mod, s = 0, newx = X[test,], exact = TRUE,x=X[train,],y=Y[train])
ridge_OLS_testmse = mean((ridge.pred_OLS-Y[test])^2)

tabledf = data.frame(
  Criterion = c("Post-RLASSO","RLASSO (default)","Post-LASSO 10-fold CV", "LASSO 10-fold CV (user)", "LASSO 10-fold CV 1 SE (user)"
                ,"Ridge 10-fold CV (user)", "Ridge 10-fold CV 1 SE (user)", "Forward Stepwise Selection", "OLS"), 
  Lambda = c("-", "-", cvpostlasso_minlambda, lasso_bestlambda, lasso_minlambda_1se, ridge_bestlambda, ridge_minlambda_1se,"-", "0"), 
  TestMSE = c(postrlasso_testmse,rlasso_testmse,cvpostlasso_testmse,lasso_cv10fold_testmse, lasso_cv10fold_1se_testmse, ridge_cv10fold_testmse, 
              ridge_cv10fold_1se_testmse, 131507.709571066, ridge_OLS_testmse)
  )

print.data.frame(tabledf)

## Table Results
##
##                      Criterion           Lambda  TestMSE
## 1                  Post-RLASSO                - 131865.5
## 2             RLASSO (default)                - 128672.8
## 3        Post-LASSO 10-fold CV 24.7707635599171 131254.7
## 4      LASSO 10-fold CV (user) 13.7742017391105 128123.7
## 5 LASSO 10-fold CV 1 SE (user) 21.3781888203595 128194.1
## 6      Ridge 10-fold CV (user) 243.783219517401 130395.1
## 7 Ridge 10-fold CV 1 SE (user) 464.548280968782 128359.0
## 8   Forward Stepwise Selection                - 131507.7
## 9                          OLS                0 193945.1

