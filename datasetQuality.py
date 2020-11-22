"""
Data set quality assessment - basic visualization and observations

Usage: Update working directory and chose the config file with the data set. Define the input vectors
from the variables available in the config file.
Input: config file and input matrices
Output: Various plots

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 27.09.2020
"""

# Import configuration file that determines the dataset to plot
#from concRaw_config import *
from concNoZero_config import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

# -------------------------------------------------------
# Define input and output matrices that are to be used for plots
xIn = X_stand
yIn = y_fromStand

# -------------------------------------------------------
# Initial matrix plots to access the data

# ref: https://matplotlib.org/tutorials/introductory/sample_plots.html#sphx-glr-tutorials-introductory-sample-plots-py
#      https://matplotlib.org/api/axes_api.html?
#      https://matplotlib.org/3.1.1/tutorials/intermediate/gridspec.html


# All attributes vs. output
nrows = math.floor(M/2)
ncolumns = math.floor(M/4)
fig1, ax1 = plt.subplots(nrows, ncolumns, figsize = (16, 16), tight_layout = True)
count = 0 # For flat indexing 

for i in range(nrows):
    for j in range(ncolumns):
        for c in range(C):
            
            # Make plots
            ax1[i, j].plot(xIn[y_class.squeeze()==c, count], yIn[y_class.squeeze()==c, 0], '.', alpha=.5)
                
        # Plot periphery and accomodate for the specificity of the Ages attribute
        ax1[i, j].grid()
        
        if count == 7: 
            ax1[i, j].set_xlabel("{0}, day".format(attributeNames[count]), fontsize=14)
        else:
            ax1[i, j].set_xlabel("{0}, kg/m^3".format(attributeNames[count]), fontsize=14)
            
        ax1[i, j].set_ylabel("{0}, MPa".format(outputAttribute), fontsize=10)
        
        count += 1
    
plt.legend(classNames, loc = 'lower right', fontsize=14)
plt.show()

# Attributes vs attributes
nrows = math.floor(M)
ncolumns = math.floor(M)
fig1, ax1 = plt.subplots(nrows, ncolumns, figsize = (16, 16), tight_layout = True)

for i in range(nrows):
    for j in range(ncolumns):
        for c in range(C):
            
            # Make plots
            ax1[i, j].plot(xIn[y_class.squeeze()==c, j], xIn[y_class.squeeze()==c, i], '.', alpha=.5)
               
        # Plot periphery
        ax1[i, j].grid()
        
        if i == 7:
            ax1[i, j].set_xlabel("{0}".format(attributeNames[j]), fontsize=12)
        if j == 0:    
            ax1[i, j].set_ylabel("{0}".format(attributeNames[i]), fontsize=12)

plt.show()

# -------------------------------------------------------
# Boxplot - to access the outliers

nrows = math.floor(M/4)
ncolumns = math.floor(M/2)
fig2, ax2 = plt.subplots(nrows, ncolumns, figsize = (16, 16), tight_layout = True)
red_diamond = dict(markerfacecolor='r', marker='D') # Marking the outliers
count = 0 # For flat indexing 

for i in range(nrows):
    for j in range(ncolumns):
    
        ax2[i, j].boxplot(xIn[:, count ], flierprops = red_diamond)
        
        # Plot periphery and accomodate for the specificity of the Ages attribute
        ax2[i, j].set_xticks([]) # Removes the ticks
        ax2[i, j].set_xlabel(attributeNames[count], fontsize=16)
        
        if count == 7: 
            ax2[i, j].set_ylabel("Days", fontsize=16)
        else:
            ax2[i, j].set_ylabel("kg/m^3", fontsize=16)
            
        count += 1
        
plt.show()


# -------------------------------------------------------
# Histogram of the attributes

from scipy import stats

plt.figure(figsize=(15,7))
for i in range(M):
    plt.subplot(2,4,i+1)
    
    # Plot normalized histograms = area under the curve must sum up to 1
    plt.hist(xIn[:,i], color=(0.2, 0.8-i*0.1, 0.4), bins = 20, density = True, stacked = True)
    
     # Plot periphery and accomodate for the specificity of the Ages attribute       
    if i == 7:
        plt.xlabel("{0}, day".format(attributeNames[i]), fontsize=16)    
    else:
        plt.xlabel("{0}, kg/m^3".format(attributeNames[i]), fontsize=16)

    # Over the histogram, plot the theoretical probability distribution function:
    x_nDist = np.linspace(xIn[:, i].min(), xIn[:, i].max(), xIn.shape[0])
    nDist = stats.norm.pdf(x_nDist, np.mean(x_nDist), np.std(x_nDist))
    plt.plot(x_nDist, nDist,'.',color='red')

plt.show()

# -------------------------------------------------------
# Plotting zero values in 2D plot

zero_mask = (xIn[:, ]==0)
fig3, ax3 = plt.subplots(figsize=(16,16))
ax3.set_title("Zero values in the dataset", fontsize=30)
ax3.imshow(zero_mask, aspect = "auto", cmap='gnuplot')
ax3.set_xticks(np.arange(np.shape(xIn)[1]))  
ax3.set_xticklabels(attributeNames, fontsize=12)
plt.show() 



# -------------------------------------------------------
# Covariance matrix and correlation matrices

#Create a common X and y array to access the correlation of ALL variables
XandY = np.concatenate((xIn, yIn), axis = 1)

# Create covariance and correlation heat maps
heatMapLabels = np.array(["At1", "At2", "At3", "At4", "At5", "At6", "At7", "At8", "Out"])

import seaborn as sn
plt.figure(figsize=(15,7))
covMatrix = np.cov(XandY.T, bias=True)
covMatrix = np.around(covMatrix, 2)
plt.title("Covariance matrix", fontsize=20)

sn.heatmap(covMatrix, cmap='hot', annot=True, annot_kws={"size": 20}, fmt='g', xticklabels=heatMapLabels, yticklabels=heatMapLabels)
plt.show()

import seaborn as sn
plt.figure(figsize=(15,7))
corrMatrix = np.corrcoef(XandY.T)
corrMatrix = np.around(corrMatrix, 2)
plt.title("Correlation matrix", fontsize=20)

sn.heatmap(corrMatrix, cmap='hot', annot=True, annot_kws={"size": 20}, fmt='g', xticklabels=heatMapLabels, yticklabels=heatMapLabels)
plt.show()




