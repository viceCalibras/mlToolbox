"""
Description: This script includes the functions used to regularize :
                - The Linear Regression model: rlr_validate() function
                - The Multinominal Logistic Regression model: regmultinominal_regression() function

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340
Created: 01.11.2020
"""
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from featureTransform import x_add_features
import sklearn.linear_model as lm


def rlr_validate(xIn, yIn, lambdas, cvf):
    
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        xIn       training data set MUST BE STANDARDIZED!
        yIn       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    
    CV = model_selection.KFold(cvf, shuffle=True)
    M = xIn.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    yIn = yIn.squeeze()
    
    for train_index, test_index in CV.split(xIn,yIn):
        
        X_train = xIn[train_index]
        y_train = yIn[train_index]
        X_test = xIn[test_index]
        y_test = yIn[test_index]
        
        # Precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        for l in range(0,len(lambdas)):
            
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power( y_train-X_train @ w[:,f,l].T, 2).mean(axis=0)
            test_error[f,l] = np.power( y_test-X_test @ w[:,f,l].T, 2).mean(axis=0)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda


def regmultinominal_regression(xIn, yIn, lambdas, cvf):
    
    CV = model_selection.KFold(cvf, shuffle=True)
    M = xIn.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    yIn = yIn.squeeze()
    
    for train_index, test_index in CV.split(xIn,yIn):
        
        X_train = xIn[train_index]
        y_train = yIn[train_index]
        X_test = xIn[test_index]
        y_test = yIn[test_index]
        
        for l in range(0,len(lambdas)):

            # Fit multinomial logistic regression model
            mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                               tol=1e-4, random_state=1, 
                                               penalty='l2', C=1/lambdas[l], max_iter=1000)
            mdl.fit(X_train,y_train)
            
            # Evaluate training and test performance
            train_error[f,l] = np.sum(mdl.predict(X_train)!=y_train) / len(y_train)
            test_error[f,l] = np.sum(mdl.predict(X_test)!=y_test) / len(y_test)
            w[:,f,l] = mdl.coef_[0] # Nice...
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda

