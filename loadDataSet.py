""""
Data load & fundemntal pre-processing script - Concrete Compressive Strength
ref: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength, accessed: 02.10.2020.

Usage: Update working directory and check that the data set is there.
Input: Concrete_Data.xls
Output: Attribute and response values and name matrices, type: numpy.array

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718 
Created: 20.09.2020

"""

import numpy as np
import pandas as pd
import xlrd
import os

# Change into working directiory & load the excell spreadsheet
os.chdir(os.path.dirname(os.path.realpath(__file__)))
doc = xlrd.open_workbook('Concrete_Data.xls').sheet_by_index(0)

# Extrapolate attribute names
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=9)

# Clean attribute names - remove everything in brackets, brackets and empty spaces
# ref: https://stackoverflow.com/questions/14596884/remove-text-between-and-in-python
import re
import string

regex = re.compile(".*?\((.*?)\)")  

for i in range(len(attributeNames)):
    
    attrClean = attributeNames[i]
    result = re.findall(regex, attrClean) # Creates a list of substrings in the brackets
    
    # Remove what is inside the brackets
    for j in range(len(result)):
        
        attrClean = attrClean.replace(result[j], "")
        
    # Remove the brackets & empty spaces   
    attrClean = attrClean.replace(")", "")
    attrClean = attrClean.replace("(", "")
    attrClean = attrClean.strip()
    
    attributeNames[i] = attrClean

# Separate input from the output   
outputAttribute = attributeNames[-1]
del attributeNames[-1]

# Preallocate memory, then extract data to matrix X and y 
row_count = doc.nrows # Needs to be modified to be used for 0 indexing
column_count = doc.ncols

X = np.empty((row_count-1, column_count-1))
y = np.empty((row_count-1, 1))

# Create data matrix
for i in range(row_count-1):
    X[i,:] = np.array(doc.row_values(i+1,0,column_count-1))


# Create output matrix    
y = np.array(doc.col_values(column_count-1,1,row_count)).T

# To avoid column vector, create N x 1 matrix
y = y.reshape(np.shape(y)[0],1)

# Compute the value N
N = len(y)






