"""
Description: This script computes and compares different feature transformations for Linear Regression problem. 
             The model with the lowest estimated generalization error - RMSE will be used as a dataset for the rest of the project.

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340
Created: 01.11.2020
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.linear_model as lm
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid
from CrossValidation import twoLevelCV_single, twoLevelCV_single_PCA
from featureTransform import x_add_features, x_tilda_poly, x_tilda_transform, x_tilda_downSample

def compare_models(X_stand, y_fromStand, modelsToCompare):
    
    # -------------------------------------------------------
    # Initialize comparison relevant parameters
    model = lm.LinearRegression()
    K1 = 10
    K2 = 10
    
    K3 = 10 # Number of total comparison loops
    modelErrors = np.zeros((K3, len(modelsToCompare)))
    
    for i in range(K3):
        
        # -------------------------------------------------------
        # Compute error for the regular model
        xIn, yIn =  X_stand, y_fromStand
        modelErrors[i, 0] = twoLevelCV_single(xIn, yIn, model, K1, K2)
        
        # -------------------------------------------------------
        # Compute error for the 6 PCA model
        xIn, yIn =  X_stand, y_fromStand
        modelErrors[i, 1] = twoLevelCV_single_PCA(xIn, yIn, model, K1, K2)
        
        # -------------------------------------------------------
        # Compute error for the added features model
        xIn, yIn =  x_add_features(X_stand, y_fromStand)
        modelErrors[i, 2] = twoLevelCV_single(xIn, yIn, model, K1, K2)
        
        # -------------------------------------------------------
        # Compute error for the polynomial regression model
        xIn, yIn =  x_tilda_poly(X_stand, y_fromStand)
        modelErrors[i, 3] = twoLevelCV_single(xIn, yIn, model, K1, K2)
        
        # -------------------------------------------------------
        # Compute error for the transformed features model
        xIn, yIn =  x_tilda_transform(X_stand, y_fromStand)
        modelErrors[i, 4] = twoLevelCV_single(xIn, yIn, model, K1, K2)
        
        # -------------------------------------------------------
        # Compute error for the chosen features model
        features = np.array([1, 2])
        xIn, yIn = x_tilda_downSample(xIn, yIn, features)
        modelErrors[i, 5] = twoLevelCV_single(xIn, yIn, model, K1, K2)
        
        # -------------------------------------------------------
        # Compute error for the transfomrmed + PCA features model
        xIn, yIn =  x_tilda_transform(X_stand, y_fromStand)
        modelErrors[i, 6] = twoLevelCV_single_PCA(xIn, yIn, model, K1, K2)
    
    # MSE calculation - for plots
    # modelErrorsAvg = np.mean(modelErrors, axis = 0)
    # RMSE calculation - for plots
    modelErrorsAvg = np.sqrt(np.mean(modelErrors, axis = 0))    

    return modelErrorsAvg


# Conduct comparison
modelsToCompare = ["Basic Linear Regression", "6 PCA", "Added Features", "Polynomial regression", "Transformed features", "Chosen features", "Transformed + PCA"]

for i in range(2):
    
    if i == 0:
        # Import configuration file that determines the dataset to be used
        from concRaw_config import *
        # Define basic parameters
        X_stand = X_stand
        y_fromStand = y_fromStand
        
        modelErrorsAvg_Raw = compare_models(X_stand, y_fromStand, modelsToCompare)
        
    else:
        # Import configuration file that determines the dataset to be used
        from concNoZero_config import *
        # Define basic parameters
        X_stand = X_stand
        y_fromStand = y_fromStand
        
        modelErrorsAvg_NoZero = compare_models(X_stand, y_fromStand, modelsToCompare)

# Plot the results
        
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = modelErrorsAvg_Raw
bars2 = modelErrorsAvg_NoZero
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
plt.bar(r1, bars1, color='darkred', width=barWidth, edgecolor='white', label='concRaw')
plt.bar(r2, bars2, color='cornflowerblue', width=barWidth, edgecolor='white', label='concNoZero')
 
# Periphery
plt.title('Regression model choice', fontsize = 24)
plt.xlabel('Models', fontsize = 24)
plt.xticks([r + barWidth/2 for r in range(len(bars1))], modelsToCompare, fontsize = 24)
plt.ylabel('Estimated generalization error - RMSE', fontsize = 24)
 
# Create legend & Show graphic
plt.legend(fontsize = 24)
plt.show()

print("Model based upon concRaw dataset with the smallest E_gen of {0} is the {1} model.".format(round(modelErrorsAvg_Raw.min(), 2), modelsToCompare[modelErrorsAvg_Raw.argmin()]))
print("Model based upon concNoZero dataset with the smallest E_gen of {0} is the {1} model.".format(round(modelErrorsAvg_NoZero.min(), 2), modelsToCompare[modelErrorsAvg_NoZero.argmin()]))




