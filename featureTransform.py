"""
Feature Engineering

Description: This script includes the functions used to compute feature transformations on the dataset

Regression analysis
Created: 30.10.2020
"""

import sklearn.linear_model as lm
from sklearn import model_selection
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
from matplotlib.pylab import subplot, hist
import math
import numpy as np


def x_add_features(xIn, yIn):
    
    # Additional nonlinear attributes features
    Xf1 = np.power(xIn[:, 0], 2).reshape(-1,1)
    Xf2 = np.power(xIn[:, 4], 2).reshape(-1,1)
    Xf3 = np.power(xIn[:, 7], 2).reshape(-1,1)
    
    # Add the transformed features into the dataset
    xAddFeat = np.asarray(np.bmat('xIn, Xf1, Xf2, Xf3'))
    yOut = yIn # For traceability
    
    return xAddFeat, yOut 


def x_tilda_poly(xIn, yIn):
    
    # Do a feature transformation - polynomial regression
    xTildaPoly = np.zeros((xIn.shape[0], xIn.shape[1]))
    
    for i in range(xIn.shape[0]):   
        for j in range(xIn.shape[1]):
            xTildaPoly[i, j] = xIn[i, j]**(j+1)
        
    yOut = yIn # For traceability

    return xTildaPoly, yOut


def x_tilda_transform(xIn, yIn):
    
    # Do a feature transformation - trigonometry based
    xTildaTrans = np.zeros((xIn.shape[0], xIn.shape[1]))
    
    for i in range(xIn.shape[0]):   
        for j in range(xIn.shape[1]):
            if j == 0:    
                xTildaTrans[i, j] = xIn[i, j]
            elif (j % 2) == 1:
                xTildaTrans[i, j] = math.sin(xIn[i, j]*j)
            else:
                xTildaTrans[i, j] = math.cos(xIn[i, j]*j)
        
    yOut = yIn # For traceability

    return xTildaTrans, yOut


def x_tilda_downSample(xIn, yIn, features):
    
    # Consider only some features in the dataset
    xTildaDown = np.zeros((xIn.shape[0], 1))
                  
    for i in range(np.size(features)):
        x_temp = np.reshape(xIn[:, features[i]], (xIn[:, features[i]].shape[0], 1))
        xTildaDown = np.append(xTildaDown, x_temp, axis = 1 )
   
    xTildaDown = np.delete(xTildaDown, 0, 1)
    yOut = yIn # For traceability

    return xTildaDown, yOut

