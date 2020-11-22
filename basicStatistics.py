"""
This function conducts basic summary stadistics calculation.

Usage: sum_stat(X, y, attributeNames, outputAttribute)
Input: Attribute and response values and name matrices, type: numpy.array
Output: Summary statistics tabular overview, type: numpy.array

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 02.10.2020
"""
import numpy as np

def sum_stat(X, y, attributeNames, outputAttribute):
   
    # -------------------------------------------------------
    # Compute summary statistics for attributes
    # Compute value of M
    M = len(attributeNames)
    
    mean = X.mean(axis=0)
    std = X.std(ddof=1,axis=0) #the function computes N - ddof, since we're calculating the empirical std (ddof=1)
    median = np.median(X,axis=0)
    min_value = X.min(axis=0)
    max_value = X.max(axis=0)
    p25 = np.percentile(X, q=25, axis=0)
    p50 = np.percentile(X, q=50, axis=0)
    p75 = np.percentile(X, q=75, axis=0)
    
    # Round down the results
    statParam = np.array([mean,std, median, min_value, p25, p50, p75, max_value])
    statParam = np.around(statParam, 2)
    mean,std, median, min_value, p25, p50, p75, max_value = statParam
    
    statisticsName = np.array([[' ','mean','std', 'median', 'min', 'p25','p50','p75','max']])
    basic_statistics_x = np.array([attributeNames, mean,std, median, min_value, p25, p50, p75, max_value])
    basic_statistics_x = np.insert(basic_statistics_x, 0, statisticsName, axis = 1)
    
    # -------------------------------------------------------
    # Compute summary statistics for output variable
    
    mean = y.mean(axis=0)
    std = y.std(ddof=1,axis=0) 
    median = np.median(y,axis=0)
    min_value = y.min(axis=0)
    max_value = y.max(axis=0)
    p25 = np.percentile(y, q=25, axis=0)
    p50 = np.percentile(y, q=50, axis=0)
    p75 = np.percentile(y, q=75, axis=0)
    
    # Round down the results
    statParam = np.array([mean,std, median, min_value, p25, p50, p75, max_value])
    statParam = np.around(statParam, 2)
    mean,std, median, min_value, p25, p50, p75, max_value = statParam
    
    basic_statistics_y = np.array([outputAttribute, mean,std, median, min_value, p25, p50, p75, max_value])
    basic_statistics_y = basic_statistics_y.reshape(np.shape(basic_statistics_y)[0],1)
    basic_statistics_y = np.insert(basic_statistics_y, 0, statisticsName, axis = 1)

    return(basic_statistics_x, basic_statistics_y)