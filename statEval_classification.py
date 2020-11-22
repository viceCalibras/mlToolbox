"""
Statistical evaluation for Classification problem 

Description: This script is using the two layer CV output to perform statistical performance evaluation of the three classification models.
             MODELS: Regularized Multinominal Logistic Regression, ANN MultiClassification, Baseline

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340
Created: 10.11.2020
"""

from CrossValidation import twoLevelCV_regression, twoLevelCV_classification
from featureTransform import x_add_features, x_tilda_poly, x_tilda_transform, x_tilda_downSample
import sklearn.linear_model as lm
from concNoZero_config import *
import scipy.stats as st

def correlated_ttest(r, rho, alpha=0.05):
    """
    made by .....
    """
    rhat = np.mean(r)
    shat = np.std(r)
    J = len(r)
    sigmatilde = shat * np.sqrt(1 / J + rho / (1 - rho))

    CI = st.t.interval(1 - alpha, df=J - 1, loc=rhat, scale=sigmatilde)  # Confidence interval
    p = 2*st.t.cdf(-np.abs(rhat) / sigmatilde, df=J - 1)  # p-value
    return p, CI


#_______CREATE DATASET WITH ADDED FEATURES_______
xIn, yIn = x_add_features(X_stand, y_class)

# Initialize 2 layer CV parameters
K1 = 5
K2 = 5

# Values of lambda
lambdas = np.logspace(-5, 5, 20)
# Range of hidden units
hidden_units = np.array((1,3,6,8,11,15))
# Parameters for ANN training part
CV_ann = 2
n_replicates=1
max_iter=15000
tolerance = 1e-7

# Comparing with two layer Cross-Validation: linear regression ,ANN and baseline
models = ['REGULARIZED_MULTINOMINAL_REGRESSION', 'ANN_MULTICLASS', 'BASELINE_CLASSIFICATION']
error_test, outer_lambdas, outer_hidden_units, r, estimatedGenError = twoLevelCV_classification(xIn, yIn, models, K1, K2, lambdas,
                                                                                                hidden_units, CV_ann=CV_ann,
                                                                                                n_replicates=n_replicates,
                                                                                                max_iter=max_iter, tolerance = tolerance)

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K1

print('Statistical Evaluation for Classification')
print('ANN vs. RLR')
p_setupII, CI_setupII = correlated_ttest(r[:,0], rho, alpha=alpha)
print("\nP value for setup II: {0}".format(round(p_setupII, 4)))
print("CI setup II: from {0} to {1}:".format(round(CI_setupII[0], 4), round(CI_setupII[1], 4) ))

print('ANN vs. Baseline')
p_setupII, CI_setupII = correlated_ttest(r[:,1], rho, alpha=alpha)
print("\nP value for setup II: {0}".format(round(p_setupII, 4)))
print("CI setup II: from {0} to {1}:".format(round(CI_setupII[0], 4), round(CI_setupII[1], 4) ))

print('RLR vs. Baseline')
p_setupII, CI_setupII = correlated_ttest(r[:,2], rho, alpha=alpha)
print("\nP value for setup II: {0}".format(round(p_setupII, 4)))
print("CI setup II: from {0} to {1}:".format(round(CI_setupII[0], 4), round(CI_setupII[1], 4) ))
