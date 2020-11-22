"""
Process concRaw data set config file.

Usage: Import loadDataSet.py and use dedicated functions from preProcessing.py to process data.
Can be used only if the number of attributes remain constant or changed inside the flow.
Must start from RAW data set!
Input: See preProcessing.py
Output: Processed data and statistical tables. Final output depends on the process flow.

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 03.10.2020
Edited: 04.11.2020
"""

from preProcessing import *
from loadDataSet import *

# -------------------------------------------------------
# Step 1: Import RAW data
xIn = X
yIn = y

# Remove outliers
X_noOut, y_noOut, M, N, C = remove_outliers(xIn, yIn, N, attributeNames)

# -------------------------------------------------------
# Step 1 input
xIn = X_noOut
yIn = y_noOut

# -------------------------------------------------------
# Centralize and standardize the data
X_cent, X_stand, y_fromStand, M, N, C = cent_and_stand(xIn, yIn, N, attributeNames)

# -------------------------------------------------------
# Step 2 input
xIn = X_stand
yIn = y_fromStand

# -------------------------------------------------------
# Create classes
classNames, y_class, y_class_encoding, M, N, C = class_threshold (xIn, yIn, attributeNames, outputAttribute)


