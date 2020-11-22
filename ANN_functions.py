"""
Description: This script has the functions used to optain the optimal number of hidden units for:
                - ANN regression: annr_validate() function
                - ANN classification: ann_multiclass_validate() function

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340
Created: 08.11.2020
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from scipy import stats

def annr_validate(xIn, yIn, hidden_units, K, n_replicates, max_iter, tolerance):
    
    loss_fn = torch.nn.MSELoss() # MSE for regression problem 
    CV = model_selection.KFold(K, shuffle=True)
    M = xIn.shape[1]
    train_error = np.empty((K,len(hidden_units)))
    test_error = np.empty((K,len(hidden_units)))
    f = 0
    
    for (k, (train_index, test_index)) in enumerate(CV.split(xIn,yIn)):
        print('\n\tCrossvalidation ANN fold: {0}/{1}'.format(k+1,K))    
        
        X_train = torch.Tensor(xIn[train_index,:])
        y_train = torch.Tensor(yIn[train_index])
        X_test = torch.Tensor(xIn[test_index,:])
        y_test = torch.Tensor(yIn[test_index])
        
        for i in range(0,len(hidden_units)):
            
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, hidden_units[i]),
                                torch.nn.Tanhshrink(),
                                torch.nn.Linear(hidden_units[i], 1),
                                )

            print('\n\t>> Training model with {} hidden units'.format(hidden_units[i]))
            
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter,
                                                               tolerance=tolerance)
     
            print('\n\tFinal loss with {} hidden_units = {}\n'.format(hidden_units[i],np.round(final_loss,4)))
            
            # Determine estimated class labels for train and test set
            y_train_est = net(X_train)
            y_test_est = net(X_test)
        
            # Evaluate training and test performance
            se_train = (y_train_est.float()-y_train.float())**2 # squared error
            mse_train = (sum(se_train).type(torch.float)/len(y_train)).data.numpy() #mean
            se_test = (y_test_est.float()-y_test.float())**2 # squared error
            mse_test = (sum(se_test).type(torch.float)/len(y_test)).data.numpy() #mean
        
            train_error[f,i] = mse_train
            test_error[f,i] = mse_test
        f+=1
        
    opt_n_hidden_units = hidden_units[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_hidden_units = np.mean(train_error,axis=0)
    test_err_vs_hidden_units = np.mean(test_error,axis=0)

    return opt_n_hidden_units, train_err_vs_hidden_units, test_err_vs_hidden_units


def ann_multiclass_validate(xIn, yIn, C, hidden_units, K, n_replicates, max_iter, tolerance):    

    loss_fn = torch.nn.CrossEntropyLoss() # CrossEntropy Loss for multi-classification problem
    CV = model_selection.KFold(K, shuffle=True)
    M = xIn.shape[1]
    train_error = np.empty((K,len(hidden_units)))
    test_error = np.empty((K,len(hidden_units)))
    f = 0

    for (k, (train_index, test_index)) in enumerate(CV.split(xIn,yIn)):
        print('\n\tCrossvalidation ANN fold: {0}/{1}'.format(k+1,K))    

        X_train = torch.from_numpy(xIn[train_index,:]).float()
        y_train = torch.from_numpy(yIn[train_index]).long().squeeze()
        X_test = torch.from_numpy(xIn[test_index,:]).float()
        y_test = torch.from_numpy(yIn[test_index]).long().squeeze()
                
        for i in range(0,len(hidden_units)):
            # Define the model
            model = lambda: torch.nn.Sequential(torch.nn.Linear(M, hidden_units[i]), 
                                                torch.nn.Tanhshrink(),
                                                torch.nn.Linear(hidden_units[i], C), 
                                                torch.nn.Softmax(dim=1)
                                                )
            print('\n\t>> Training model with {} hidden units'.format(hidden_units[i]))           
            net, _, _ = train_neural_net(model,
                                        loss_fn,
                                        X=X_train,
                                        y=y_train,
                                        n_replicates=n_replicates,
                                        max_iter=max_iter,
                                        tolerance=tolerance)

            # Determine probability of each class using trained network
            softmax_logits_train = net(X_train)
            softmax_logits_test = net(X_test)

            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_train_est = (torch.max(softmax_logits_train, dim=1)[1]).data.numpy()
            y_test_est = (torch.max(softmax_logits_test, dim=1)[1]).data.numpy()

            # Compute error rates
            e_train = (y_train_est != y_train.data.numpy())
            e_test = (y_test_est != y_test.data.numpy())
              
            train_error[f, i] = np.sum(e_train) / len(y_test)
            test_error[f, i] = np.sum(e_test) / len(y_test)
            
            print('\n\tTest error rate ({} hidden_units) = {}'.format(hidden_units[i],np.round(test_error[f, i],4)))
            print('\tMiss-classifications ({} hidden units) = {} out of {}'.format(hidden_units[i],sum(e_test),len(e_test)))
        f+=1
    
    opt_n_hidden_units = hidden_units[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_hidden_units = np.mean(train_error,axis=0)
    test_err_vs_hidden_units = np.mean(test_error,axis=0)

    return opt_n_hidden_units, train_err_vs_hidden_units, test_err_vs_hidden_units


def train_neural_net(model, loss_fn, X, y, n_replicates=3, max_iter = 10000, tolerance=1e-6):
    """
    Args:
        model:          A function handle to make a torch.nn.Sequential.
        loss_fn:        A torch.nn-loss, e.g.  torch.nn.BCELoss() for binary 
                        binary classification, torch.nn.CrossEntropyLoss() for
                        multiclass classification, or torch.nn.MSELoss() for
                        regression (see https://pytorch.org/docs/stable/nn.html#loss-functions)
        n_replicates:   An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned.
        max_iter:       An integer specifying the maximum number of iterations
                        to do (default 10000).
        tolerenace:     A float describing the tolerance/convergence criterion
                        for minimum relative change in loss (default 1e-6)
                        
        
    Returns:
        A list of three elements:
            best_net:       A trained torch.nn.Sequential that had the lowest 
                            loss of the trained replicates
            final_loss:     An float specifying the loss of best performing net
            learning_curve: A list containing the learning curve of the best net.
    
    """
    import torch
    # Specify maximum number of iterations for training
    logging_frequency = 2000 # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        
        print('\tReplicate: {}/{}'.format(r+1, n_replicates))
    
        net = model()
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
                     
        optimizer = torch.optim.Adam(net.parameters(), lr = 0.5)
        
        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        
        for i in range(max_iter):
            
            y_est = net(X) # forward pass, predict labels on training set
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            
            if p_delta_loss < tolerance: break
        
            old_loss = loss_value
            
            # display loss with some frequency:
            if (i != 0) & ((i+1) % logging_frequency == 0):
                
                print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
                
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            
        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve