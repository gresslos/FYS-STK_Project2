# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:20:02 2024

@author: User
"""

#---------------------imports-------------------------------#
import autograd.numpy as np
from autograd import grad, elementwise_grad


import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split


from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      
from tensorflow.keras.layers import Dense           
from tensorflow.keras import optimizers             
from tensorflow.keras import regularizers, initializers           


import sys
import os

#------------------- imports from the general network implementation ----------------------#

from Network_implementation import Scaler, Network
from Network_implementation import CostOLS, CostLogReg, CostCrossEntropy
from Network_implementation import identity, sigmoid, softmax, RELU, LRELU
from Network_implementation import derivate, FrankeFunction, Design_Matrix_2D
from Network_implementation import NN_model

#training data
x = np.linspace(0, 1, 21)
y = np.linspace(0, 1, 21)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)
x_flat = x.flatten()
y_flat = y.flatten()
z = z.flatten().reshape(-1,1)

X_ = np.vstack((x_flat,y_flat)).T
X = Design_Matrix_2D(1, X_)



# Scaling the data for the tensorflow and analytical cases, as it is not
#automatically handled, like in our network implementation
scaler1 = Scaler(classification = False) #scaler for TF data

scaler2 = Scaler(classification = False) #scaler for analytical data

X_TFscaledtrain, z_TFscaledtrain = scaler1.scaletrain(X, z)



#testing data
np.random.seed(2024)
x_t = np.random.uniform(0, 1, size = 10)
y_t = np.random.uniform(0, 1, size = 10)
x_t, y_t = np.meshgrid(x_t,y_t)
z_t = FrankeFunction(x_t, y_t)
x_tflat = x_t.flatten()
y_tflat = y_t.flatten()
z_t = z_t.flatten().reshape(-1,1)

X__t = np.vstack((x_tflat,y_tflat)).T
X_t = Design_Matrix_2D(1, X__t)



X_TFscaledtest, z_TFscaledtest = scaler1.scaletest(X_t, z)


#----------------------------- Optimal fit for own implementation --------------------#
MLP_own = Network([2,100,1],RELU,identity,CostOLS,"XAVIER")

MLP_own.fit(X, z, n_batches = 10, n_epochs = 100, eta = 0.1, lmb = 1e-5, delta_mom = 0, method = "Adam")

MSEown = MLP_own.predict(X, z, X_t,z_t)[0]

print(f"Optimal fit for our implementation wielded MSE = {MSEown} for L.R. of 0.1 and L2 reg. of 1e-5")
#---------------------------- Analytical result -----------------------#
analyticalMSE = MSEown + 1
i = 0
while analyticalMSE > MSEown:
    i += 1
    
    X_analytical = Design_Matrix_2D(i, X_) #the analytical approach uses a design matrix
    #of degree 7
    X_scaledtrain, z_scaledtrain = scaler2.scaletrain(X_analytical, z)
    
    X_tanalytical = Design_Matrix_2D(i, X__t)
    
    X_scaledtest, z_scaledtest = scaler2.scaletest(X_tanalytical, z_t)

    OLSbeta = np.linalg.pinv(X_scaledtrain.T @ X_scaledtrain) @ X_scaledtrain.T @ z_scaledtrain

    z_tildeOLS = X_scaledtest @ OLSbeta

    z_predOLS = scaler2.rescale(z_tildeOLS)

    analyticalMSE = mean_squared_error(z_t, z_predOLS)
    
    print(f"Degree {i}, MSE {analyticalMSE}")
    
print(f"An OLS analytical linear regression to a degree {i} polynomial wielded an MSE of {analyticalMSE}")
#-------------------------- Optimal fit for TF implementation -----------------------#
eta_vals = np.logspace(-5, 0, 6)
lmbd_vals = np.logspace(-5, 0, 6)
lmbd_vals = np.append(lmbd_vals, 0)

MSEsTF = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        print(f"Eta: {eta}, lambda: {lmbd}")
        MLP_TF = NN_model(2,1,100,"ADAM", eta, lmbd, 'relu', "XAVIER")
        MLP_TF.fit(X_TFscaledtrain, z_TFscaledtrain, epochs=100, batch_size=int(X.shape[0]/10), verbose=0)
        z_tilde = MLP_TF.predict(X_TFscaledtest)
        z_pred = scaler1.rescale(z_tilde)
        MSEsTF[i][j] = mean_squared_error(z_t, z_pred)
#Finding the optimal MSE and corresponding eta and lambda values for TensorFlow
optimalMSETF, indexoptimalTF = np.min(MSEsTF), np.unravel_index(np.argmin(MSEsTF), MSEsTF.shape)
optimaletaTF, optimalmbdTF = eta_vals[indexoptimalTF[0]], lmbd_vals[indexoptimalTF[1]]

print(f"TensorFlow obtained an optimal MSE of {optimalMSETF} for L.R. of {optimaletaTF} and L2 reg. of {optimalmbdTF}")

