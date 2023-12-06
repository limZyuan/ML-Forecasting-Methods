rm(list=ls()) #use this to clear the variables

##### Applied #####
library(wooldridge)


### Section 1 ###
help(wage2)

## 1.
#wage: monthly earnings
#hours: average weekly hours
#IQ: IQ score
#KWW: knowledge of world work score
#educ: years of education
#exper: years of work experience
#tenure: years with current employer
#age: age in years
#married: =1 if married
#black: =1 if black
#south: =1 if live in south
#urban: =1 if live in SMSA
#sibs: number of siblings
#brthord: birth order
#meduc: mother's education
#feduc: father's education
#lwage: natural log of wage

## 2. 
#I would think the following variables are important in the prediction of wage:
#hours, IQ, KWW, educ, exper, tenure, age, married, black, south, urban, sibs,
#brthord, meduc, feduc
#lwage is excluded as it does not contribute to the prediction of wage. Including it 
#could instead make things worse.



### Section 2 ###
df = wage2

## 1. 
#Remove rows that contain at least one missing value
df = na.omit(df)

## 2. 
#Set the training set size:
trsize=500

#seed of the random number generator for replicability
set.seed(2457829)

#randomly generate trsize numbers from 1 to the nrow(df)
trindices = sample(1:nrow(df),trsize)

#get the training sample (getting only rows whose indices are in trindices)
train = df[trindices,]

#get the testing sample (remove rows whose indices are in trindices)
test = df[-trindices,]



### Section 3 ###
library(leaps)

## 1.
#This is the formula used in `regsubsets()`
fml = formula(wage~poly(hours,degree = 3,raw = TRUE)
              +poly(IQ,degree = 3,raw = TRUE)
              +poly(KWW,degree = 3,raw = TRUE)
              +poly(educ,degree = 3,raw = TRUE)
              +poly(exper,degree = 3,raw = TRUE)
              +poly(tenure,degree = 3,raw = TRUE)
              +married
              +black
              +south
              +urban
              +poly(sibs,degree = 3,raw = TRUE)
              +poly(brthord,degree = 3,raw = TRUE)
              +poly(meduc,degree = 3,raw = TRUE)
              +poly(feduc,degree = 3,raw = TRUE))

test_mat = model.matrix(fml, data = test)
#34 regressors

## 2.
maxregno = 20 #Max number of regressors
bss = regsubsets(fml, data = train, nvmax = maxregno, method = "exhaustive")
bss_sum = summary(bss)

#Finding BIC and AIC for "variance estimate from the model"
#Find which position in `bic` is the lowest `bic`.
#This position corresponds to the `k` (number of variables).
kbic=which.min(bss_sum$bic) 

#Find which position in `aic` is the lowest `aic`.
#This position corresponds to the `k` (number of variables).
kaic=which.min(bss_sum$cp)


#Finding BIC and AIC for "variance from the largest model" when P << N
#estimate error variance of the model with k=20
errvar= bss_sum$rss[maxregno]/(trsize - maxregno - 1) #trsize-maxregno-1 because there are 20 regressors and 1 constant

#construct the IC using `errvar`, _man is for 'manual'
bic_man = bss_sum$rss/trsize + log(trsize)*errvar*((1:maxregno)/trsize)
aic_man = bss_sum$rss/trsize + 2*errvar*((1:maxregno)/trsize)

#BIC choice
kbic_man=which.min(bic_man)
#AIC choice
kaic_man=which.min(aic_man)


#Finding BIC and AIC for "iterative variance" when when P is relative large w.r.t N
i = 1 #i denotes number of iteration
varYbic = var(train$wage) #sample variance of Y
varYaic = var(train$wage) #sample variance of Y
kbic_temp = 0; kaic_temp = 0
repeat{
  if (i > 1){ ## For iteration after the first, we need to update the error variance
    varYbic = bss_sum$rss[kbic_iter]/(trsize-kbic_iter-1) 
    varYaic = bss_sum$rss[kaic_iter]/(trsize-kaic_iter-1)         
  }
  
  bic_iter = bss_sum$rss/trsize + log(trsize)*varYbic*((1:maxregno)/trsize)
  aic_iter = bss_sum$rss/trsize + 2*varYaic*((1:maxregno)/trsize) 
  #Select best models
  kbic_iter = which.min(bic_iter)
  kaic_iter = which.min(aic_iter)
  if (kbic_temp == kbic_iter & kaic_temp == kaic_iter){
    break #when convergence occurs, stop the loop
  }
  #store this iteration s' choice of k to compare with the new ones later.
  kbic_temp = kbic_iter
  kaic_temp = kaic_iter 
  i = i +1
}

#In general, all 3 methods for AIC picked a more complex model (k = 12) as compared to all 3
#methods of BIC which picked a less complex model (k = 6).
#The reasoning for this is because for this training size at trsize = 500, the model complexity penalty
#term for BIC is higher than AIC.

## 3.

#Display a message about best choices:
cat('Best subset selection','\n')
cat('BIC best k: ',kbic ,'\n')
cat('AIC best k: ',kaic ,'\n')
cat('BIC best k - variance from biggest model: ',kbic_man,'\n')
cat('AIC best k - variance from biggest model: ',kaic_man,'\n')
cat('BIC best k - variance from iterative variance: ',kbic_iter,'\n')
cat('AIC best k - variance from iterative variance: ',kaic_iter,'\n')



### Section 4 ###

## 1.
#BIC
#OOS MSE from from all 3 methods
chosen_coefs = coef(bss, id = kbic)
bic_vars = names(chosen_coefs)
predicted = test_mat[,bic_vars]%*%chosen_coefs
truevals = test$wage
bic_mse = mean((truevals-predicted)^2)


#AIC:
#OOS MSE from all 3 methods
chosen_coefs = coef(bss, id = kaic) 
aic_vars = names(chosen_coefs)
predicted = test_mat[,aic_vars]%*%chosen_coefs
truevals = test$wage
aic_mse = mean((truevals-predicted)^2)



### Section 5 ###

## 1.1 
fml_expand = formula(wage~poly(cbind(hours,IQ, KWW, educ, exper, tenure,sibs, brthord,meduc,feduc), degree = 3,raw=TRUE)
                     +married
                     +black
                     +south
                     +urban)

test_mat_for = model.matrix(fml_expand, data = test)
#289 regressors

## 1.2. 
maxregno_bfs = 20
bfs = regsubsets(fml_expand, data = train, nvmax = maxregno_bfs, method = "forward")
bfs_sum = summary(bfs)

#Finding BIC and AIC for "variance estimate from the model"
#Information criteria 
kbic_for=which.min(bfs_sum$bic) #BIC choice
kaic_for=which.min(bfs_sum$cp)  #AIC choice (AIC proportional to Cp)


#Finding BIC and AIC for "variance from the largest model" when P << N
#estimate error variance of the model with k=20
errvar_for =bfs_sum$rss[maxregno_bfs]/(trsize - maxregno_bfs - 1) 

#construct the IC with this estimate: 
## (NOTE: #man: for manual; for: for forward_
bic_man_for = bfs_sum$rss/trsize + log(trsize)*errvar_for*((1:maxregno_bfs)/trsize)
aic_man_for = bfs_sum$rss/trsize + 2*errvar_for*((1:maxregno_bfs)/trsize)

kbic_man_for=which.min(bic_man_for) #BIC choice

kaic_man_for=which.min(aic_man_for) #AIC choice (AIC proportional to Cp and so ranking is the same)


#Finding BIC and AIC for "iterative variance" when when P is relative large w.r.t N
i = 1 #i denotes number of iteration
varYbic_for = var(train$wage) #sample variance of Y
varYaic_for = var(train$wage) #sample variance of Y
kbic_temp_for = 0; kaic_temp_for = 0
repeat{
  if (i > 1){ ## For iteration after the first, we need to update the error variance
    varYbic_for = bfs_sum$rss[kbic_iter_for]/(trsize-kbic_iter_for-1) 
    varYaic_for = bfs_sum$rss[kaic_iter_for]/(trsize-kaic_iter_for-1)
    ### NOTE: (trsize-kaic_iter_for-1) is the degree of freedom 
    #### (accounted for how many variables are involved)
  }
  
  bic_iter_for = bfs_sum$rss/trsize + 
    log(trsize)*varYbic_for*((1:maxregno_bfs)/trsize)
  aic_iter_for = bfs_sum$rss/trsize + 
    2*varYaic_for*((1:maxregno_bfs)/trsize) 
  #Select best models
  kbic_iter_for = which.min(bic_iter_for)
  kaic_iter_for = which.min(aic_iter_for)
  if (kbic_temp_for == kbic_iter_for & kaic_temp_for == kaic_iter_for){
    break #when convergence occurs, stop the loop
  }
  #store this iteration s' choice of k to compare with the new ones later.
  kbic_temp_for = kbic_iter_for
  kaic_temp_for = kaic_iter_for
  i = i +1
}

#In general, all 3 methods for AIC picked a more complex model (k = 17 and K = 19) as compared to all 3
#methods of BIC which picked a less complex model (k = 5).
#The reasoning for this is because for this training size at trsize = 500, the model complexity penalty
#term for BIC is higher than AIC.

## 1.3.

#Display a message about best choices:
cat('Forward stepwise selection','\n')
cat('BIC best k: ',kbic_for ,'\n')
cat('AIC best k: ',kaic_for ,'\n')
cat('BIC best k - variance from biggest model: ',kbic_man_for,'\n')
cat('AIC best k - variance from biggest model: ',kaic_man_for,'\n')
cat('BIC best k - variance from iterative variance: ',kbic_iter_for,'\n')
cat('AIC best k - variance from iterative variance: ',kaic_iter_for,'\n')


## 1.4.
#BIC
#OOS MSE from from all 3 methods
chosen_coefs = coef(bfs, id = kbic_for)
bic_vars = names(chosen_coefs)
predicted = test_mat_for[,bic_vars]%*%chosen_coefs
truevals = test$wage
bic_mse_for_5 = mean((truevals-predicted)^2)


#AIC:
#OOS MSE from "variance estimate from the model"
chosen_coefs = coef(bfs, id = kaic_for) 
aic_vars = names(chosen_coefs)
predicted = test_mat_for[,aic_vars]%*%chosen_coefs
truevals = test$wage
aic_mse_for_17 = mean((truevals-predicted)^2)

#OOS MSE from other 2 methods
chosen_coefs = coef(bfs, id = kaic_man_for) 
aic_vars = names(chosen_coefs)
predicted = test_mat_for[,aic_vars]%*%chosen_coefs
truevals = test$wage
aic_mse_for_19 = mean((truevals-predicted)^2)

## 2.
#Question: Can we perform best subset selection in step 9? Explain why or why not.
#Answer:
#No we cannot. This is due to computational reasons where it is not viable to apply best subset selection on very large p.
#The search space is very large with a number of 2^289 possible models in this case. Forward stepwise selection which explores
#a far more restricted set of models (1+(289(289+1)/2 = 41,906 models in this case) is much more attractive.

#Question: Why is the test MSE of the model chosen by AIC is bigger than the test MSE of the model chosen by BIC, 
#even though the number of regressors chosen by AIC is bigger than that chosen by BIC?
#Answer: 
#By increasing the number of regressors of the model,it only reduces the in-sample MSE or training error.
#However, the test MSE is using out-of-sample wage values. The test MSE accounts for the bias-variance trade-off
#which means that there is now a penalty term for model complexity. This is why the test MSE for AIC is bigger despite 
#having more regressors as the penalty term increased more than the reduction in bias.

## 3.
#Question: Why consistently the number of regressors chosen by BIC is smaller than that chosen by AIC?
#Answer:
#The reasoning for this is because for this training size at trsize = 500, the model complexity penalty term for BIC is higher than AIC. 
#For BIC it is log(500)*k = 2.70*k while for AIC it is 2*k.
