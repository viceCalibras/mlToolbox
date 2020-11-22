"""
This script visualizes effect of different lambdas and hidden units.
Moreover, it also computes the number of classes (low, medium and high concret) that the dataset has 
after the whole preprocessing process.

Usage: adjust the inputs, run the script
Input: optimal hyperparameters for the models and CV parameters
Output: hyperparameters plots

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340
Created: 13.11.2020
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import model_selection
import sklearn.linear_model as lm
from scipy import stats
from ANN_functions import *
from concNoZero_config import *
from featureTransform import x_add_features
from regularization import rlr_validate, regmultinominal_regression


xIn,yIn = x_add_features(X_stand, y_fromStand)
M = xIn.shape[1]
attributeNames.append('Xf1')
attributeNames.append('Xf2')
attributeNames.append('Xf3')
classNames = classNames.tolist()

#%%
# Unbalanced Dataset
print("Observations of Low Concrete = {}".format(np.sum(y_class.squeeze()==0)))
print("Observations of Medium Concrete = {}".format(np.sum(y_class.squeeze()==1)))
print("Observations of High Concrete = {}".format(np.sum(y_class.squeeze()==2)))

# BASELINE CLASSIFICATION MODEL
baseline_class = np.array((np.sum(y_class.squeeze()==0), np.sum(y_class.squeeze()==1), np.sum(y_class.squeeze()==2)))
baseline_model_prediction = np.argmax(baseline_class)*np.ones(y_class.shape[0])
error_test = np.sum(baseline_model_prediction != y_class) / len(y_class)
print(error_test)

#%%
#-------- REGULARIZED LINEAR REGRESSION -------------------------
# Add offset attribute
xInReg = np.concatenate((np.ones((xIn.shape[0],1)), xIn),1)

# Parameters
lambdas = np.power(10.,np.arange(-10,10,0.5))
cvf = 10
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(xInReg, yIn, lambdas, cvf=cvf)

# Display the results for the last cross-validation fold
plt.figure(1, figsize=(12,8))
plt.subplot(1,2,1)
plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.grid()
plt.legend(attributeNames[1:], loc='best')

plt.subplot(1,2,2)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()

print("Optimal regularization strenght is: {0}".format(round(opt_lambda, 4)))


#%%
#------- REGULARIZED MUTINOMINAL REGRESSION ---------------------------
# Parameters
lambdas = np.logspace(-5, 5, 20)
cvf = 10
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = regmultinominal_regression(xIn, y_class, lambdas, cvf=cvf)

# Display the results for the last cross-validation fold
plt.figure(1, figsize=(12,8))
plt.subplot(1,2,1)
plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.grid()
plt.legend(attributeNames[1:], loc='best')

plt.subplot(1,2,2)
plt.title('Optimal lambda: 1e{0}'.format(np.round(np.log10(opt_lambda), 4)))
plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()

print("Optimal regularization strenght is: {0}".format(round(opt_lambda, 8)))


#%%
#-------- ANN REGRESSION -------------------------
# Parameters
hidden_units = np.array((1,3,6,8,11,15,20,25))
CV_ann = 2
n_replicates = 1
max_iter = 15000
tolerance = 1e-7

opt_n_hidden_units, train_err_vs_hidden_units, test_err_vs_hidden_units = annr_validate(xIn, yIn, hidden_units, CV_ann, n_replicates=n_replicates, max_iter=max_iter, tolerance = tolerance)

plt.figure(1, figsize=(8,8))
plt.title('Optimal number of hidden units: {}'.format(opt_n_hidden_units))
plt.plot(hidden_units,train_err_vs_hidden_units.T,'b.-',hidden_units,test_err_vs_hidden_units.T,'r.-')
plt.xlabel('Number of hidden units')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()
print('Optimal number of hidden units: {}'.format(opt_n_hidden_units))

#%%
#-------MULTI-CLASS ANN CLASSIFICATION ---------------------------
# Parameters
hidden_units = np.array((1,3,6,8,11,15))
CV_ann = 2
n_replicates=1
max_iter=15000
tolerance = 1e-7

opt_n_hidden_units, train_err_vs_hidden_units, test_err_vs_hidden_units = ann_multiclass_validate(xIn, y_class, C, hidden_units, CV_ann, n_replicates=n_replicates, max_iter=max_iter, tolerance = tolerance)
# torch.CrossEntropy: combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

plt.figure(1, (8,8))
plt.title('Optimal number of hidden units: {}'.format(opt_n_hidden_units))
plt.plot(hidden_units,train_err_vs_hidden_units.T,'b.-',hidden_units,test_err_vs_hidden_units.T,'r.-')
plt.xlabel('Number of hidden units')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()
print('Optimal number of hidden units: {}'.format(opt_n_hidden_units))
