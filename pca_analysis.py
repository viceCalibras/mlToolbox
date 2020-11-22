"""
PCA analysis

Usage: Update working directory and chose the config file with the data set or just use separate functions. 
Number of classes in the compared data sets must be the same!
Input: see separate functions
Output: PCA plots and output matrices
        

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 03.10.2020
Edited by: Vice, 01.11.2020 - added projected dataset export & created a plot functions
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
import numpy as np

"""
# -------------------------------------------------------
# Standard import header that determines the dataset to plot - 
# Created for convenience, commen in as neccessay
# Take care that the order of imports and variable definition is in place!

# Import 1. dataset - D1
# Define input and output matrices that are to be used for plots
from concRaw_config import *
# Define input and output matrices that are to be used for plots
xInFirst = X_stand
yInFirst = y_fromStand
yIn_classFirst = y_class

# Import 2. dataset - D2
from concNoZero_config import *
# Define input and output vectors that are to be used for plots
xInSecond = X_stand
yInSecond = y_fromStand
yIn_classSecond = y_class
"""

def pca_compute(xInFirst, xInSecond, threshold, pcUsed):
    
    """
    General PCA analysis for 2 datasets. 
    Input: xInFirst - first data set
           xInSecond - second data set
           thershold - variance explaine threshold
           pcUsed - used principal components for the plot
    Output: computed PCs, projected data, variance explained and component load.
    """
    
    # -------------------------------------------------------
    # Compute the PCA by computing SVD of D1 and D2
    
    U_D1,S_D1,Vh_D1 = svd(xInFirst, full_matrices=False)
    U_D2,S_D2,Vh_D2 = svd(xInSecond, full_matrices=False)
    
    V_D1 = Vh_D1.T
    V_D1_round = np.around(V_D1,3)
    
    V_D2 = Vh_D2.T
    V_D2_round = np.around(V_D2,3)
    
    # Compute variance explained by principal components
    rho_D1 = (S_D1*S_D1) / (S_D1*S_D1).sum()
    rho_D1_round = np.round(rho_D1,3)
    rho_D2 = (S_D2*S_D2) / (S_D2*S_D2).sum() 
    rho_D2_round = np.around(rho_D2,3)
        
    # --------------------------------------------------------------------------
    # Export the projected data in first pcUsed principal compontents
    
    # Compute the projection onto the principal components for D1 and D2
    Z_D1 = xInFirst @ V_D1
    Z_D2 = xInSecond @ V_D2
    
    Z_D1_out = Z_D1[:, :pcUsed]
    Z_D2_out = Z_D2[:, :pcUsed]
    
    return (Z_D1, Z_D2, Z_D1_out, Z_D2_out, rho_D1, rho_D2, V_D1, V_D2)

# Z_D1, Z_D2, Z_D1_out, Z_D2_out, rho_D1, rho_D2, V_D1, V_D2 = pca_compute(xInFirst, xInSecond, threshold = 0.95, pcUsed = 6 )


def pca_hist(xRaw, attributeNames):
    
    # -------------------------------------------------------
    # Histogram of the attributes standard deviation
    # Histogram is to be plotted for RAW data, to access the need for standardization
    
    plt.figure(figsize=(15,7))
    plt.bar(attributeNames,np.std(xRaw, ddof=1,axis=0))
    plt.ylabel('Standard deviation', fontsize = 16)
    plt.xlabel('Attributes', fontsize = 16)
    plt.title('STD assessment', fontsize = 16)
    plt.show()
    
    return

def pca_var_expl(rho_D1, rho_D2, threshold):
      
    #-------------------------------------------------------
    #Plot variance explained by principal components (D1)
    
    plt.figure(figsize=(15,7))
    plt.plot(range(1,len(rho_D1)+1),rho_D1,'x-')
    rho_D1_acum_round = np.around(np.cumsum(rho_D1),3)
    plt.plot(range(1,len(rho_D1)+1),np.cumsum(rho_D1),'o-')
    plt.plot([1,len(rho_D1)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components (concRaw)', fontsize = 16);
    plt.xlabel('Principal component', fontsize = 16);
    plt.ylabel('Variance explained', fontsize = 16);
    plt.legend(['Individual','Cumulative','Threshold'], fontsize = 15)
    plt.grid()
    plt.show()
    
    # -------------------------------------------------------
    #Plot variance explained by principal components (D2)
    
    plt.figure(figsize=(15,7))
    plt.plot(range(1,len(rho_D2)+1),rho_D2,'x-')
    rho_D2_acum = np.around(np.cumsum(rho_D2),3)
    plt.plot(range(1,len(rho_D2)+1),np.cumsum(rho_D2),'o-')
    plt.plot([1,len(rho_D2)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components (concNoZero)', fontsize = 16);
    plt.xlabel('Principal component', fontsize = 16);
    plt.ylabel('Variance explained', fontsize = 16);
    plt.legend(['Individual','Cumulative','Threshold'], fontsize = 15)
    plt.grid()
    plt.show()

    return

def pca_comp_load(V_D1, V_D2, attributeNames):
    
    # -------------------------------------------------------------------------------
    # Plot PCA Component Loadings (D1) 
    
    plt.figure(figsize=(22,7))
    pcs_names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']
    num_att = [0,1,2,3,4,5,6,7]
    bw = .1
    r = np.arange(1,len(pcs_names)+1)
    for i in num_att:    
        plt.bar(r+i*bw, V_D1[i,:6], width=bw)
    plt.xticks(r+bw, pcs_names)
    plt.xlabel('Component coefficients', fontsize = 16)
    plt.ylabel('Loading', fontsize = 16)
    plt.legend(attributeNames)
    plt.grid()
    plt.title('PCA Component Loading (concRaw)', fontsize = 20)
    plt.show()
    
    # -------------------------------------------------------------------------------
    # Plot PCA Component Loadings (D2) 
    
    plt.figure(figsize=(22,7))
    pcs_names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']
    num_att = [0,1,2,3,4,5,6,7]
    bw = .1
    r = np.arange(1,len(pcs_names)+1)
    for i in num_att:    
        plt.bar(r+i*bw, V_D2[i,:6], width=bw)
    plt.xticks(r+bw, pcs_names)
    plt.xlabel('Component coefficients', fontsize = 16)
    plt.ylabel('Loading', fontsize = 16)
    plt.legend(attributeNames)
    plt.grid()
    plt.title('PCA Component Loading (concNoZeros)', fontsize = 20)
    plt.show()
    
    return

def pca_comp_coeff(V_D1, V_D2, attributeNames):

    # --------------------------------------------------------------------------
    # Plot relevant PCA Component Coefficients (D1) - first n PC
    
    plt.figure(figsize=(22,9))
    pcs = [0,1,2,3,4,5]
    legendStrs = ['PC'+str(e+1) for e in pcs]
    bw = .1
    r = np.arange(1,M+1)
    
    for i in pcs: 
        plt.bar(r+i*bw, V_D1[:,i], width=bw) 
        
    plt.xticks(r+bw, attributeNames)
    plt.xlabel('Attributes', fontsize = 16)
    plt.ylabel('Component coefficients', fontsize = 16)
    plt.legend(legendStrs)
    plt.grid()
    plt.title('PCA Component Coefficients (concRaw)', fontsize = 20)
    plt.show()
    
    # --------------------------------------------------------------------------
    # Plot PCA Component Coefficients (D2) - first n PC
    
    plt.figure(figsize=(22,7))
    pcs = [0,1,2,3,4,5]
    legendStrs = ['PC'+str(e+1) for e in pcs]
    bw = .1
    r = np.arange(1,M+1)
    
    for i in pcs:    
        plt.bar(r+i*bw, V_D2[:,i], width=bw)
        
    plt.xticks(r+bw, attributeNames)
    plt.xlabel('Attributes', fontsize = 16)
    plt.ylabel('Component coefficients', fontsize = 16)
    plt.legend(legendStrs)
    plt.grid()
    plt.title('PCA Component Coefficients (concNoZero)', fontsize = 20)
    plt.show()

    return

def pca_analysis(xInFirst, xInSecond, yIn_classFirst, yIn_classSecond, threshold):
    
    #--------------------------------------------------------------------------
    # PCA analysis overview plot
    
    # Store the two data sets in a cell  as Y1 and Y2, so it can be traversed
    Ys = [xInFirst, xInSecond]
    titles = ['concRaw', 'concNoZero']
    
    # Choose two PCs to plot (the projection)
    i = 0
    j = 1
    
    # Make the plot
    plt.figure(figsize=(10,15))
    plt.subplots_adjust(hspace=.4)
    plt.title('Concrete Compressive Strength: Effect of standardization', fontsize = 20)
    nrows=3
    ncols=2
    
    for k in range(2):
        
        # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
        U,S,Vh = svd(Ys[k],full_matrices=False)
        V=Vh.T # For the direction of V to fit the convention in the course we transpose Hermitian operator
        
        # For visualization purposes, we flip the directionality of the
        # principal directions such that the directions match for Y1 and Y2.
        if k==1: V = -V; U = -U; 
        
        # Compute variance explained
        rho = (S*S) / (S*S).sum() 
        
        # Compute the projection onto the principal components
        Z = Ys[k] @ V;
        
        # Plot projection
        plt.subplot(nrows, ncols, 1+k)
           
        for c in range(C):
            
            if k == 0:
                plt.plot(Z[yIn_classFirst.squeeze()==c, i], Z[yIn_classFirst.squeeze()==c, j], '.', alpha=.5)
            else:
                plt.plot(Z[yIn_classSecond.squeeze()==c, i], Z[yIn_classSecond.squeeze()==c, j], '.', alpha=.5)
                
            
        plt.xlabel('PC'+str(i+1), fontsize = 12)
        plt.ylabel('PC'+str(j+1), fontsize = 12)
        plt.title(titles[k] + '\n' + 'Projection', fontsize = 12 )
        plt.legend(classNames)
        plt.axis('equal')
        
        
        # Plot attribute coefficients in principal component space
        plt.subplot(nrows, ncols,  3+k)
        for att in range(V.shape[1]):
            plt.arrow(0,0, V[att,i], V[att,j])
            plt.text(V[att,i], V[att,j], attributeNames[att])
            
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.xlabel('PC'+str(i+1), fontsize = 12)
        plt.ylabel('PC'+str(j+1), fontsize = 12)
        plt.grid()
        
        # Add a unit circle
        plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
             np.sin(np.arange(0, 2*np.pi, 0.01)));
        
        plt.title(titles[k] +'\n'+'Attribute coefficients', fontsize = 12)
        plt.axis('equal')
                
        # Plot cumulative variance explained
        plt.subplot(nrows, ncols,  5+k);
        plt.plot(range(1,len(rho)+1),rho,'x-')
        plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
        plt.plot([1,len(rho)],[threshold, threshold],'k--')
        plt.title('Variance explained by principal components', fontsize = 12);
        plt.xlabel('Principal component', fontsize = 12);
        plt.ylabel('Variance explained', fontsize = 12);
        plt.legend(['Individual','Cumulative','Threshold'])
        plt.grid()
        plt.title(titles[k]+'\n'+'Variance explained', fontsize = 12)
    
    plt.show()
    
    return

def pca_3D_plot(Z_D1, Z_D2, yIn_classFirst, yIn_classSecond, C):

    # --------------------------------------------------------------------------
    # Plot 3D projection on 3 principal components
    
    pcs = [0,1,2]
    
    # 3D scatter plot for D1 projected in the first 3 PC
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    
    for c in range(C):
        
        class_mask = yIn_classFirst.squeeze()==c
        s = ax.scatter(Z_D1[class_mask,pcs[0]], Z_D1[class_mask,pcs[1]], Z_D1[class_mask,pcs[2]])
        
    ax.view_init(30, 220)
    ax.set_xlabel('PC'+str(pcs[0]+1), fontsize = 12)
    ax.set_ylabel('PC'+str(pcs[1]+1), fontsize = 12)
    ax.set_zlabel('PC'+str(pcs[2]+1), fontsize = 12)
    ax.set_title('Projection of standardized data in the first 3 PC - concRaw ', fontsize = 12)
    plt.show()
    
    
    # 3D scatter plot for D2 projected in the first 3 PC
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    
    for c in range(C):
        
        class_mask = yIn_classSecond.squeeze()==c
        s = ax.scatter(Z_D2[class_mask,pcs[0]], Z_D2[class_mask,pcs[1]], Z_D2[class_mask,pcs[2]])
        
    ax.view_init(30, 220)
    ax.set_xlabel('PC'+str(pcs[0]+1), fontsize = 12)
    ax.set_ylabel('PC'+str(pcs[1]+1), fontsize = 12)
    ax.set_zlabel('PC'+str(pcs[2]+1), fontsize = 12)
    ax.set_title('Projection of the standardized data in the first 3 PC - concNoZeros', fontsize = 12)
    plt.show()
    
    return


