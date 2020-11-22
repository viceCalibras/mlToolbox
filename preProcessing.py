"""
This function conducts data set pre-processing - centralization, standardization, 
outlier removal, thresholding and removal of zero values.

Usage: function_name(see specific function for required arguments)
Input: See specific functions
Output: Various processed attribute and response values and name matrices, type: numpy.array

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 02.10.2020
Edited: 04.11.2020
"""

from basicStatistics import *
import numpy as np
    
# -------------------------------------------------------
# Centralize and standardize the data
    
def cent_and_stand(X, y, N, attributeNames):

    # Create the centered data
    X_cent = X - np.ones((N,1))*X.mean(0) # Mean along the axis   
    
    # Create standardized data - equivalent to the z-score
    # Subtract the mean from the data and divide by the attribute standard deviation
    X_stand = X_cent*(1/np.std(X_cent,0))
    
    # Compute values of N and C
    N = len(y)
    C = np.shape(y)[1]
    # Compute value of M
    M = len(attributeNames)
    
    y_fromStand = y # For traceability
    
    return (X_cent, X_stand, y_fromStand, M, N, C)
        
# -------------------------------------------------------
# Thresholding the output variable - Concrete Compressive Strenght
# Thersholds are set according to the summary statistics results
    
def class_threshold (X, y, attributeNames, outputAttribute):

    basic_statistics_x, basic_statistics_y = sum_stat(X, y, attributeNames, outputAttribute)
    
    lowStrThreshold = basic_statistics_y[5, 1]
    medStrThreshold = basic_statistics_y[7, 1]
    
    y_lowMask = (y < lowStrThreshold)
    y_medMask = (y > lowStrThreshold) & (y < medStrThreshold)
    y_highMask = (y > medStrThreshold)
    
    classNames = np.asarray(("Low Strenght Concrete", "Medium Strenght Concrete", "High Strenght Concrete"))
    classDict = dict(zip(classNames, range(len(classNames))))
        
    # Create class output vector
    y_class = np.zeros(y.shape[0])
    y_class = y_class.reshape(y.shape[0], 1)
    y_class = y_class.astype(int)
    y_class[y_lowMask] = classDict["Low Strenght Concrete"]
    y_class[y_medMask] = classDict["Medium Strenght Concrete"]
    y_class[y_highMask] = classDict["High Strenght Concrete"]
    
    # Do 1-out-of K encoding
    K = int(y_class.max()+1)
    y_class_encoding = np.zeros((y_class.size, K))
    y_class_encoding[np.arange(y_class.size), y_class[:, 0]] = 1
    
    # Compute values of N and C
    N = len(y)
    C = len(classNames)
    # Compute value of M
    M = len(attributeNames)
    
    return(classNames, y_class, y_class_encoding, M, N, C)
    
# -------------------------------------------------------
# Remove zero values - to access the batches that have all the ingredients

def remove_zeros(X, y, attributeNames):
    
    X_zeroMask = (X == 0)
    X_nonZeroRows = np.logical_not(np.any(X_zeroMask, axis = 1))
    
    X_noZeros = np.array(X[X_nonZeroRows, :])
    y_noZeros = np.array(y[X_nonZeroRows, :])
    
    # Compute values of N and C
    N = len(y_noZeros)
    C = np.shape(y_noZeros)[1]
    # Compute value of M
    M = len(attributeNames)
    
    return(X_noZeros, y_noZeros, M, N, C)
    
    
# -------------------------------------------------------
# Clean the outliers

def remove_outliers(X, y, N, attributeNames):
    
    # Outliers are defined as those outside of 1.5 x IQR + Q3 / - Q1 range ()
    Q3 = np.percentile(X, 75, axis = 0)
    Q1 = np.percentile(X, 25, axis = 0)
    IQR_1p5 = np.ones((N,1)) * ((Q3 - Q1) * 1.5)
    
    outMax = Q3 + IQR_1p5
    outMin = Q1 - IQR_1p5
    
    X_outMaxMask = (X >= outMax)
    X_outMinMask = (X <= outMin)
    
    X_OutMask = X_outMaxMask + X_outMinMask
    X_noOutRows = np.logical_not(np.any(X_OutMask, axis = 1))
    
    X_noOut = np.array(X[X_noOutRows, :])
    y_noOut = np.array(y[X_noOutRows, :])
    
    # Compute values of N and C
    N = len(y_noOut)
    C = np.shape(y_noOut)[1]
    # Compute value of M
    M = len(attributeNames)
        
    return(X_noOut, y_noOut, M, N, C)




