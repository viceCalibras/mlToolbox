"""
Cross-Validation

Description: This script performs Two level cross-validation for model selection and performace evaluation 
             (based on algorithm 6, lecture notes)

             - twoLevelCV_regression() function -> Compares the three regression models using two level Cross-Validation
             - twoLevelCV_classification() function -> Compares the three classification models using two level Cross-Validation

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340
Created: 07.11.2020
"""

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np
import torch

from pca_analysis import pca_compute
from regularization import rlr_validate, regmultinominal_regression
from ANN_functions import *


# -------------------------------------------------------
# Compare the three regression models using two level Cross-Validation

def twoLevelCV_regression(xIn, yIn, models, K1, K2, lambdas, hidden_units, CV_ann, n_replicates, max_iter, tolerance):
    
    M = xIn.shape[1]
    CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)
    
    # Initialize variables
    error_train = np.empty((K2, len(models)))
    error_val = np.empty((K2, len(models)))
    
    error_test = np.empty((K1, len(models)))
    
    inner_lambdas = np.zeros(K2) # Inner loop values for optimal lambda
    outer_lambdas = np.zeros(K1) # Outer loop values for optimal lambda

    inner_hidden_units = np.zeros(K2) # Inner loop values for optimal number of hidden units
    outer_hidden_units = np.zeros(K1) # Outer loop values for optimal number of hidden units 

    best_models_idx = np.empty((1, len(models)))
    estimatedGenError = np.empty((1, len(models)))
    
    # r parameter for the correlated t test initialization
    r = np.empty((K1, len(models)))
    
    # Outer cross-validation loop. Performance Evaluation
    k1 = 0 
    for par_index, test_index in CV_outer.split(xIn):
        
        # extract par and test set for current CV fold
        X_par = xIn[par_index, :]
        y_par = yIn[par_index]
        X_test = xIn[test_index, :]
        y_test = yIn[test_index]
        
        # Inner cross-validation loop. Model selection and parameter optimization
        k2 = 0
        models_rlr = []
        models_ann = []
        models_baseline = []
        
        for train_index, val_index in CV_inner.split(X_par):
            print("\nOuter Iteration {0}/{1} -----------------------------".format(k1+1, K1))
            print("\nInner Iteration {0}/{1} -----------------------------".format(k2+1, K2))
            
            # Extract train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
    
                        
            for s, model in enumerate(models):
                
                if s == 0: # REGULARIZED LINEAR REGRESSION
                    
                    print("\nInner {}/{} - Regularized Linear Regression".format(k2+1,K2))
                    # Add offset attribute - to accomodate for the specificity of the rlr_validate code
                    xInReg = np.concatenate((np.ones((X_train.shape[0],1)), X_train),1)
                    opt_lambda = rlr_validate(xInReg, y_train, lambdas, 10)[1]
                    # Save the values of the optimal regularization strength
                    inner_lambdas[k2] = opt_lambda

                    print("Optimal Lambda = {}".format(np.round(opt_lambda, 4)))
                    modelRLR = lm.Ridge(alpha=opt_lambda)
                    m = modelRLR.fit(X_train, y_train)

                    # Save the trained model
                    models_rlr.append(m)

                    # Compute MSEs
                    error_train[k2, s] = np.square( y_train - m.predict(X_train) ).sum() / y_train.shape[0]
                    error_val[k2, s] = np.square( y_val - m.predict(X_val) ).sum() / y_val.shape[0]

                if s==1: # ANN REGRESSION
                    
                    print("\nInner {}/{} - ANN Regression".format(k2+1,K2))

                    opt_n_hidden_units = annr_validate(X_train, y_train, hidden_units, CV_ann, n_replicates=n_replicates, max_iter=max_iter, tolerance=tolerance)[0]
                    inner_hidden_units[k2] = opt_n_hidden_units
                    
                    # Training the ann model with the optimal number of hidden units
                    
                    model = lambda: torch.nn.Sequential(torch.nn.Linear(M, opt_n_hidden_units),
                                                        torch.nn.Tanhshrink(),
                                                        torch.nn.Linear(opt_n_hidden_units, 1),
                                                        )
                    

                    # Training the ann model with the optimal number of hidden units
                    print("\n\tTraining the model with the optimal number of hidden units")
                    loss_fn = torch.nn.MSELoss()
                    net = train_neural_net( model,
                                            loss_fn,
                                            X=torch.Tensor(X_train),
                                            y=torch.Tensor(y_train),
                                            n_replicates=n_replicates,
                                            max_iter=max_iter)[0]
                    # Save the trained model
                    models_ann.append(net)
                    
                    # Compute MSEs
                    y_train_est = net(torch.Tensor(X_train))
                    y_val_est = net(torch.Tensor(X_val))
                    
                    error_train[k2, s] = np.square( y_train - y_train_est.data.numpy() ).sum() / y_train.shape[0]
                    error_val[k2, s] = np.square( y_val - y_val_est.data.numpy() ).sum() / y_val.shape[0]
            
                if s==2: # BASELINE REGRESSION
                
                    print("\nInner {}/{} - Baseline Regression".format(k2+1,K2))
                    baseline_model = np.mean(y_train)
                    models_baseline.append(baseline_model)
                    # Compute MSEs
                    error_val[k2, s] = np.square( y_val - baseline_model ).sum() / y_val.shape[0]                    
                    
                
                print("Validation error - Model {0}: {1}".format(s+1, np.round(error_val[k2, s], 4) ))
                
            k2 += 1
            
        print("\nSummary Optimal models Outer {}/{}".format(k1+1,K1))
        for s, model in enumerate(models): 
            
            # Find the CV index of optimal models
            best_models_idx[0, s] = error_val[:, s].argmin()
            print("\n- The best model {0} was: CV number {1}".format(s+1, int(best_models_idx[0, s]+1)))
            
            if s == 0: # Save the optimal lambda of the optimal model
                
                # Trace back the model according to its CV fold index
                modelrlr_opt = models_rlr[int(best_models_idx[0, s])]

                # Compute MSE for the optimal model
                error_test[k1, s] = np.square( y_test - modelrlr_opt.predict(X_test) ).sum()/y_test.shape[0]
                outer_lambdas[k1] = inner_lambdas[int(best_models_idx[0, s])]
            
            if s==1: # Save the optimal number of hidden units of the optimal model
                
                # Trace back the model according to its CV fold index
                net_opt = models_ann[int(best_models_idx[0, s])]
                                    
                # Compute MSE for the optimal model
                y_test_est = net_opt(torch.Tensor(X_test))
                error_test[k1, s] = np.square( y_test - y_test_est.data.numpy() ).sum()/y_test.shape[0]
                outer_hidden_units[k1] = inner_hidden_units[int(best_models_idx[0, s])]
        
            if s==2: # Baseline computing test error
                
                # Trace back the model according to its CV fold index
                modelbaseline_opt = models_baseline[int(best_models_idx[0, s])]
                # Compute MSE for the baseline
                error_test[k1, s] = np.square( y_test - modelbaseline_opt ).sum() / y_test.shape[0]
            
        # Append the list of the differences in the generalization errors of two models. r is a matrix of k1 rows and 3 columns 
        # column 0: ann vs lrl - column 1: ann vs baseline - column 2: lrl vs baseline  (same notation as the project description)
        r[k1,0] = np.mean(error_test[:, 1]) - np.mean(error_test[:, 0])
        r[k1,1] = np.mean(error_test[:, 1]) - np.mean(error_test[:, 2])
        r[k1,2] = np.mean(error_test[:, 0]) - np.mean(error_test[:, 2])
        
        k1 += 1
        print("\n")
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    print("\n")
    for s in range(len(models)):
        print("Estimated Generalization Error for Model {0}: {1}".format(s+1, estimatedGenError[s]))

    return error_test, outer_lambdas, outer_hidden_units, r, estimatedGenError 


# -------------------------------------------------------
# Compare the three classification models using two level Cross-Validation

def twoLevelCV_classification(xIn, yIn, models, K1, K2, lambdas, hidden_units, CV_ann, n_replicates, max_iter, tolerance):

    M = xIn.shape[1]
    CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)
    
    # Initialize variables
    error_train = np.empty((K2, len(models)))
    error_val = np.empty((K2, len(models)))
    error_test = np.empty((K1, len(models)))
    
    inner_lambdas = np.zeros(K2) # Inner loop values for optimal lambda
    outer_lambdas = np.zeros(K1) # Outer loop values for optimal lambda

    inner_hidden_units = np.zeros(K2) # Inner loop values for optimal number of hidden units
    outer_hidden_units = np.zeros(K1) # Outer loop values for optimal number of hidden units

    best_models_idx = np.empty((1, len(models)))
    estimatedGenError = np.empty((1, len(models)))
    
    # r parameter for the correlated t test initialization
    r = np.empty((K1, len(models)))
    
    # Outer cross-validation loop. Performance Evaluation
    k1 = 0 
    for par_index, test_index in CV_outer.split(xIn):
        
        # extract par and test set for current CV fold
        X_par = xIn[par_index, :]
        y_par = yIn[par_index]
        X_test = xIn[test_index, :]
        y_test = yIn[test_index]
        
        # Inner cross-validation loop. Model selection and parameter optimization
        k2 = 0
        models_rmr = []
        models_ann = []
        models_baseline = []
        
        for train_index, val_index in CV_inner.split(X_par):
            print("\nOuter Iteration {0}/{1} -----------------------------".format(k1+1, K1))
            print("\nInner Iteration {0}/{1} -----------------------------".format(k2+1, K2))
            
            # Extract train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
    
                        
            for s, model in enumerate(models):
                
                if s == 0: # REGULARIZED MULTINOMINAL LOGISTIC REGRESSION
                    
                    print("\nInner {}/{} - Regularized Multinominal Regression".format(k2+1,K2))

                    opt_lambda = regmultinominal_regression(xIn, yIn, lambdas, cvf=10)[1]
                    # Save the values of the optimal regularization strength
                    inner_lambdas[k2] = opt_lambda
                    print("Optimal Lambda = {}".format(np.round(opt_lambda,3)))
                    # Fit multinomial logistic regression model
                    modelRMR = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                               tol=1e-4, random_state=1, 
                                               penalty='l2', C=1/opt_lambda, max_iter=1000)
                    m = modelRMR.fit(X_train, y_train)

                    # Save the trained model
                    models_rmr.append(m)

                    # Compute Errors Rate = {number of misclassified observations}/len(y_val)
                    error_train[k2, s] = np.sum(m.predict(X_train)!=y_train) / len(y_train)
                    error_val[k2, s] = np.sum(m.predict(X_val)!=y_val) / len(y_val)

                if s==1: # ANN MULTI-CLASSIFICATION
                    
                    print("\nInner {}/{} - ANN MultiClassification".format(k2+1,K2))
                    
                    opt_n_hidden_units = ann_multiclass_validate(X_train, y_train, 3, hidden_units, CV_ann, n_replicates=n_replicates, max_iter=max_iter, tolerance=tolerance)[0]
                    inner_hidden_units[k2] = opt_n_hidden_units

                    model = lambda: torch.nn.Sequential(torch.nn.Linear(M,opt_n_hidden_units), 
                                                        torch.nn.Tanhshrink(),
                                                        torch.nn.Linear(opt_n_hidden_units, 3), 
                                                        torch.nn.Softmax(dim=1)
                                                        )
                    
                    # Training the ann model with the optimal number of hidden units
                    print("\n\tTraining the model with the optimal number of hidden units")
                    loss_fn = torch.nn.CrossEntropyLoss()
                    net = train_neural_net( model,
                                            loss_fn,
                                            X=torch.from_numpy(X_train).float(),
                                            y=torch.from_numpy(y_train).long().squeeze(),
                                            n_replicates=n_replicates,
                                            max_iter=max_iter,
                                            tolerance=tolerance)[0]
                    # Save the trained model
                    models_ann.append(net)
                    
                    # Determine probability of each class using trained network
                    softmax_logits_train = net(torch.from_numpy(X_train).float())
                    softmax_logits_val = net(torch.from_numpy(X_val).float())
                    
                    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
                    y_train_est = (torch.max(softmax_logits_train, dim=1)[1]).data.numpy()
                    y_val_est = (torch.max(softmax_logits_val, dim=1)[1]).data.numpy()
                    
                    # Compute Errors Rate = {number of misclassified observations}/len(y_val)
                    e_train = (y_train_est != y_train)
                    e_val = (y_val_est != y_val)
              
                    error_train[k2, s] = np.sum(e_train) / len(y_train)
                    error_val[k2, s] = np.sum(e_val) / len(y_val)
            
                if s==2: # BASELINE CLASSIFICATION
                
                    print("\nInner {}/{} - Baseline Classification".format(k2+1,K2))
                    baseline_class = np.array((np.sum(y_train.squeeze()==0), np.sum(y_train.squeeze()==1), np.sum(y_train.squeeze()==2)))
                    models_baseline.append(baseline_class)
                                        
                    # Compute Errors Rate = {number of misclassified observations}/len(y_val)
                    baseline_prediction = np.argmax(baseline_class)*np.ones(y_val.shape[0]) 
                    error_val[k2, s] = np.sum( (baseline_prediction != y_val) ) / len(y_val)                                       
                
                print("Validation error - Model {0}: {1}".format(s+1, np.round(error_val[k2, s], 4) ))
                
            k2 += 1
            
        print("\nSummary Optimal models Outer {}/{}".format(k1+1,K1))
        for s, model in enumerate(models): 
            
            # Find the CV index of optimal models
            best_models_idx[0, s] = error_val[:, s].argmin()
            print("\n- The best model {0} was: CV number {1}".format(s+1, int(best_models_idx[0, s]+1)))
            
            
            if s == 0: # Save the optimal lambda of the optimal model

                # Trace back the model according to its CV fold index
                modelrmr_opt = models_rmr[int(best_models_idx[0, s])]

                # Compute Error Test Rate for the optimal model
                error_test[k1, s] = np.square( y_test - modelrmr_opt.predict(X_test) ).sum()/y_test.shape[0]   
                outer_lambdas[k1] = inner_lambdas[int(best_models_idx[0, s])]
            
            if s==1: # Save the optimal number of hidden units of the optimal model
                
                # Trace back the model according to its CV fold index
                net_opt = models_ann[int(best_models_idx[0, s])]
                                    
                # Compute Error Test Rate for the optimal model
                softmax_logits_test = net_opt(torch.from_numpy(X_test).float())
                y_test_est = (torch.max(softmax_logits_test, dim=1)[1]).data.numpy()
                error_test[k1, s] = np.sum((y_test_est != y_test)) / len(y_test)
                outer_hidden_units[k1] = inner_hidden_units[int(best_models_idx[0, s])]
        
            if s==2: # Baseline computing test error
                
                # Trace back the model according to its CV fold index
                modelbaseline_opt = models_baseline[int(best_models_idx[0, s])]
                
                # Compute Error Test Rate for the optimal baseline
                baseline_prediction_opt = np.argmax(modelbaseline_opt)*np.ones(y_test.shape[0])
                error_test[k1, s] = np.sum(baseline_prediction_opt != y_test) / len(y_test)
            
        # Append the list of the differences in the generalization errors of two models. r is a matrix of k1 rows and 3 columns 
        # column 0: ann vs lrl - column 1: ann vs baseline - column 2: lrl vs baseline  (same notation as the project description)
        r[k1,0] = np.mean(error_test[:, 1]) - np.mean(error_test[:, 0])
        r[k1,1] = np.mean(error_test[:, 1]) - np.mean(error_test[:, 2])
        r[k1,2] = np.mean(error_test[:, 0]) - np.mean(error_test[:, 2])
        
        k1 += 1
        print("\n")
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    print("\n")
    for s in range(len(models)):
        print("Estimated Generalization Error for Model {0}: {1}".format(s+1, estimatedGenError[s]))

    return error_test, outer_lambdas, outer_hidden_units, r, estimatedGenError


# -------------------------------------------------------
# Compute E_gen for a single model

def twoLevelCV_single(xIn, yIn, model, K1, K2):
    
    '''
    Input: (numpy array) xIn matrix, (numpy array) yIn matrix, (module) model,
            (int) K1:folds in outer loop, (int) K2:folds in inner loop, 
    Output: (numpy array) estimatedGenError
    '''
    
    CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)
    
    # Initialize variables
    error_test = np.empty(K1)
    error_train = np.empty(K2)
    error_val = np.empty(K2)
    
    # Outer cross-validation loop. Performance Evaluation
    k1 = 0 
    for par_index, test_index in CV_outer.split(xIn):
        
        print("\nOuter Iteration {0}/{1} -----------".format(k1+1, K1))
        # extract par and test set for current CV fold
        X_par = xIn[par_index, :]
        y_par = yIn[par_index]
        X_test = xIn[test_index, :]
        y_test = yIn[test_index]
        trainSetsX = []
        trainSetsY = [] 
                
        # Inner cross-validation loop. Model Selection
        k2 = 0
        
        for train_index, val_index in CV_inner.split(X_par):
            
            print("\nInner Iteration {0}/{1}".format(k2+1, K2))
            
            # Extract train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
    
            trainSetsX.append(X_train) # To trace back optimal models
            trainSetsY.append(y_train)
            
            m = model.fit(X_train, y_train)
             
            # Compute MSEs
            error_train[k2] = np.square( y_train - m.predict(X_train) ).sum() / y_train.shape[0]
            error_val[k2] = np.square( y_val - m.predict(X_val) ).sum() / y_val.shape[0]
             
            print("Validation error {0}:".format(np.round(error_val[k2], 4) ))
                
            k2 += 1
        
        # Trace back the model according to its CV fold index               
        print("Inner CV fold of the best model for the last loop: {0}".format(error_val.argmin()+1))
        m = model.fit(trainSetsX[error_val.argmin()], trainSetsY[error_val.argmin()])

        # Compute MSE
        error_test[k1] = np.square( y_test - m.predict(X_test) ).sum()/y_test.shape[0]
        
        k1 += 1
        
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    print("\n")
    print("Estimated Generalization Error: {0}".format(estimatedGenError))

    return estimatedGenError


# Define PCA parameters
threshold = 0.95 
pcUsed = 6
# -------------------------------------------------------
# Compute E_gen for a single model that is using PCA acquired features
    
def twoLevelCV_single_PCA(xIn, yIn, model, K1, K2):
    
    '''
    NO NEED TO INPUT PCA TRANSFORMED DATA!
    Input: (numpy array) xIn matrix, (numpy array) yIn matrix,(module) model,
            (int) K1:folds in outer loop, (int) K2:folds in inner loop, 
    Output: (numpy array) estimatedGenError
    '''
    
    CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)
    
    # Initialize variables
    error_test = np.empty(K1)
    error_train = np.empty(K2)
    error_val = np.empty(K2)
    
    # Outer cross-validation loop. Performance Evaluation
    k1 = 0 
    for par_index, test_index in CV_outer.split(xIn):
        
        print("\nOuter Iteration {0}/{1} -----------".format(k1+1, K1))
        # extract par and test set for current CV fold
        X_par = xIn[par_index, :]
        y_par = yIn[par_index]
        X_test = xIn[test_index, :]
        y_test = yIn[test_index]
        trainSetsX = []
        trainSetsY = []
                
        # Inner cross-validation loop. Model Selection
        k2 = 0
        
        for train_index, val_index in CV_inner.split(X_par):
            
            print("\nInner Iteration {0}/{1}".format(k2+1, K2))
            
            # Extract train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
            
            trainSetsX.append(X_train) # To trace back optimal models
            trainSetsY.append(y_train)
            
            # Extract projected data set and PCA space vector, fit the model
            V_D_temp = pca_compute(X_train, X_train, threshold, pcUsed)[6]
            X_train_PCA = pca_compute(X_train, X_train, threshold, pcUsed)[2]
            
            m = model.fit(X_train_PCA, y_train)
                 
            # Project validation data into the training PCA space, compute MSE
            X_val_temp = X_val @ V_D_temp
            X_val_temp = X_val_temp[:, :pcUsed]
             
            # Compute MSEs
            error_train[k2] = np.square( y_train - m.predict(X_train_PCA) ).sum() / y_train.shape[0]
            error_val[k2] = np.square( y_val - m.predict(X_val_temp) ).sum() / y_val.shape[0]
             
            print("Validation error {0}:".format(np.round(error_val[k2], 4) ))
                
            k2 += 1
        
        # Trace back the model according to its CV fold index               
        print("Inner CV fold of the best model for the last loop: {0}".format(error_val.argmin()+1))
                   # Extract projected data set and PCA space vector, fit the model
        X_temp = trainSetsX[error_val.argmin()]
        y_temp = trainSetsY[error_val.argmin()]
        
        V_D_temp = pca_compute(X_temp, y_temp, threshold, pcUsed)[6]
        X_temp = pca_compute(X_temp, X_temp, threshold, pcUsed)[2]
        
        m = model.fit(X_temp, y_temp)
        
        X_test_PCA = X_test @ V_D_temp
        X_test_PCA = X_test_PCA[:, :pcUsed]
        
        # Compute MSE        
        error_test[k1] = np.square( y_test - m.predict(X_test_PCA) ).sum()/y_test.shape[0]
        
        k1 += 1
        
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    print("\n")
    print("Estimated Generalization Error: {0}".format(estimatedGenError))

    return estimatedGenError