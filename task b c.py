# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:16:47 2024

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

import warnings
warnings.simplefilter("error")
import sys
import os

#------------------- imports from the general network implementation ----------------------#

from Network_implementation import Scaler, Network
from Network_implementation import CostOLS, CostLogReg, CostCrossEntropy
from Network_implementation import identity, sigmoid, softmax, RELU, LRELU
from Network_implementation import derivate, FrankeFunction, Design_Matrix_2D

#------------------------------------- ARCHITECTURE CHOICES -----------------------------#

hiddenact = int(input("Which hidden activation function do you wish to use? \n Press 0 for sigmoid \n Press 1 for RELU \n Press 2 for LRELU\n"))
if hiddenact == 0:
    activation = sigmoid
    
elif hiddenact == 1:
    activation = RELU
    
elif hiddenact == 2:
    activation = LRELU

else:
    sys.exit()
    
initi = int(input("How do you wish to initialize the parameters? \n Press 0 for N(0,1) \n Press 1 for HE \n Press 2 for XAVIER\n"))
if initi == 0:
    initialization = "RANDN"
    
elif initi == 1:
    initialization = "HE"
    
elif initi == 2:
    initialization = "XAVIER"
else:
    sys.exit()
    
# Where to save the figures and data files (week 37 lecture notes)
PROJECT_ROOT_DIR = "Results"



if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

#------------------------------------ ANALYSIS CHOICES -----------------------------#

dataset = int(input("Which dataset do you wish to use? \n Press 0 for 1D data \n Press 1 for Franke's Function\n"))

if dataset == 0:
    # Create a folder to store results for 1D data (if it doesn't exist)
    DATASET_ID = "Results/1D data"
    if not os.path.exists(DATASET_ID):
        os.makedirs(DATASET_ID)
    
    data = "1D" #For naming the output files
    np.random.seed(2024)
    n = 1000
    X = 2*np.random.rand(n,1)
    z = 2 + X + 2*X**2 + np.random.randn(n,1)
    
elif dataset == 1:
    # Create a folder to store results for Franke's function data (if it doesn't exist)
    DATASET_ID = "Results/Franke data"
    if not os.path.exists(DATASET_ID):
        os.makedirs(DATASET_ID)
    
    
    data = "Franke"
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y)
    x_flat = x.flatten()
    y_flat = y.flatten()
    z = z.flatten().reshape(-1,1)
    
    X_ = np.vstack((x_flat,y_flat)).T
    X = Design_Matrix_2D(1, X_)

else:
    sys.exit() #ends the program if an invalid option is selected
    
analysis_type = int(input("Which type of analysis do you wish to perform? \n Press 0 for parameter grid search \n Press 1 for Bootstrap\n"))

if analysis_type == 0:
    """
    Creating the folders for storing the generated data (if they don't exist already)
                                                         
    Differentiated first by the type of analysis, then by the chosen hidden activation,
    and finally by the initialization scheme.
    
    Having chosen all of these, they are then split up into the different gradient
    descent methods (though this is a default option if grid search is selected)
    """
    ANALYSIS_ID = os.path.join(DATASET_ID, "Grid search")
    if not os.path.exists(ANALYSIS_ID):
        os.makedirs(ANALYSIS_ID)
    
    ACTIVATION_ID = os.path.join(ANALYSIS_ID, f"{activation.__name__}")
    if not os.path.exists(ACTIVATION_ID):
        os.makedirs(ACTIVATION_ID)
        
    INITIALIZATION_ID = os.path.join(ACTIVATION_ID, f"{initialization}")
    if not os.path.exists(INITIALIZATION_ID):
        os.makedirs(INITIALIZATION_ID)
    
    MSE_PATH = os.path.join(INITIALIZATION_ID, "MSE plots")
    if not os.path.exists(MSE_PATH):
        os.makedirs(MSE_PATH)
        
    R2_PATH = os.path.join(INITIALIZATION_ID, "R2 plots")
    if not os.path.exists(R2_PATH):
        os.makedirs(R2_PATH)
    
    eta_vals = np.logspace(-5, 0, 6)
    lmbd_vals = np.logspace(-5, 0, 6)
    lmbd_vals = np.append(lmbd_vals, 0)


    # grid search
    MSEsSGD = np.zeros( ( len(eta_vals), len(lmbd_vals) ) ) 
    MSEsAdagrad = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    MSEsRMSprop = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    MSEsADAM = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    
    R2sSGD = np.zeros( ( len(eta_vals), len(lmbd_vals) ) ) 
    R2sAdagrad = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    R2sRMSprop = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    R2sADAM = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    """
    Store parameter settings that returned negative R2 values, which will be hidden 
    in the heatmaps. That way, no information is lost for the report.
    """
    lostinfoSGD, lostinfoAdagrad, lostinfoRMSprop, lostinfoADAM = list(), list(), list(), list()
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            print(f"Eta: {eta}, lambda: {lmbd}")
            
            MLP = Network([X.shape[1],100,1], activation, identity, CostOLS, initialization) #MLP with 1 hidden layer with 100 neurons
            MLP.set_classification()
            MLP.reset_weights()
            
            try:
                MSESGD, R2SGD = MLP.fit(X, z, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'SGD')[:-1]
                if R2SGD < 0:
                    lostinfoSGD.append((eta,lmbd, R2SGD))
                MSEsSGD[i][j] = MSESGD
                R2sSGD[i][j] = R2SGD
            except RuntimeWarning:
                MSEsSGD[i][j]=float("inf")
                R2sSGD[i][j]=float("inf")
                pass;
            
            MLP.reset_weights()
            try:    
                MSEAdagrad, R2Adagrad = MLP.fit(X, z, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'Adagrad')[:-1]
                if R2Adagrad < 0:
                    lostinfoAdagrad.append((eta,lmbd, R2Adagrad))
                MSEsAdagrad[i][j] = MSEAdagrad
                R2sAdagrad[i][j] = R2Adagrad
            except RuntimeWarning:
                MSEsAdagrad[i][j]=float("inf")
                R2sAdagrad[i][j]=float("inf")
                pass;   
         
            MLP.reset_weights()
            try:    
                MSERMSprop, R2RMSprop = MLP.fit(X, z, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'RMSprop')[:-1]
                if R2RMSprop < 0:
                    lostinfoRMSprop.append((eta,lmbd, R2RMSprop))
                MSEsRMSprop[i][j] = MSERMSprop
                R2sRMSprop[i][j] = R2RMSprop
            except RuntimeWarning:
                MSEsRMSprop[i][j]=float("inf")
                R2sRMSprop[i][j]=float("inf")
                pass;   
            
            MLP.reset_weights()    
            try:    
                MSEADAM, R2ADAM = MLP.fit(X, z, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'Adam')[:-1]
                if R2ADAM < 0:
                    lostinfoADAM.append((eta,lmbd, R2ADAM))
                MSEsADAM[i][j] = MSEADAM
                R2sADAM[i][j] = R2ADAM
                print(MSEADAM)
            except RuntimeWarning:
                MSEsADAM[i][j]=float("inf")
                R2sADAM[i][j] = float("inf")
                pass;
    
    # ------------------------------------- STORING OPTIMAL MODELS --------------------------- #
    # Optimal MSE and corresponding indices
    optimalMSESGD, indexoptimalSGD = np.min(MSEsSGD), np.unravel_index(np.argmin(MSEsSGD), MSEsSGD.shape)
    optimalMSEAdagrad, indexoptimalAdagrad = np.min(MSEsAdagrad), np.unravel_index(np.argmin(MSEsAdagrad), MSEsAdagrad.shape)
    optimalMSERMSprop, indexoptimalRMSprop = np.min(MSEsRMSprop), np.unravel_index(np.argmin(MSEsRMSprop), MSEsRMSprop.shape)
    optimalMSEADAM, indexoptimalADAM = np.min(MSEsADAM), np.unravel_index(np.argmin(MSEsADAM), MSEsADAM.shape)
    
    # "Optimal" R2 (in principle, the highest R2 should correspond to the lowest MSE,
    # either way, MSE takes a higher priority in our analysis) 
    optimalR2SGD = R2sSGD[indexoptimalSGD[0]][indexoptimalSGD[1]]
    optimalR2Adagrad = R2sAdagrad[indexoptimalAdagrad[0]][indexoptimalAdagrad[1]]
    optimalR2RMSprop = R2sRMSprop[indexoptimalRMSprop[0]][indexoptimalRMSprop[1]]
    optimalR2ADAM = R2sADAM[indexoptimalADAM[0]][indexoptimalADAM[1]]
    
    optimaletaSGD, optimalmbdSGD = eta_vals[indexoptimalSGD[0]], lmbd_vals[indexoptimalSGD[1]]
    optimaletaAdagrad, optimalmbdAdagrad = eta_vals[indexoptimalAdagrad[0]], lmbd_vals[indexoptimalAdagrad[1]]
    optimaletaRMSprop, optimalmbdRMSprop = eta_vals[indexoptimalRMSprop[0]], lmbd_vals[indexoptimalRMSprop[1]]
    optimaletaADAM, optimalmbdADAM = eta_vals[indexoptimalADAM[0]], lmbd_vals[indexoptimalADAM[1]]

    # Plot heatmaps with inf values highlighted (mean squared error)
    fig, ax = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsSGD)
    sns.heatmap(np.log10(MSEsSGD), annot=True, ax=ax, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'MSE (log 10)'},annot_kws={"size": 12})
    cbar = ax.collections[0].colorbar
    cbar.set_label('MSE (log 10)', fontsize=15)
    ax.set_title(f"MSE for SGD, {activation.__name__} activation, {initialization} scheme", fontsize=20)
    ax.set_ylabel("Learning rate", fontsize = 15)
    ax.set_xlabel("Lambda", fontsize = 15)
    ax.add_patch(plt.Rectangle((indexoptimalSGD[1], indexoptimalSGD[0]), 1, 1, fill=False, edgecolor='orange', lw=3))
    plt.savefig(MSE_PATH + f"/MSE_SGD_own_{activation.__name__}_{initialization}_{data}.png")
    plt.show()
    
    
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsAdagrad)
    sns.heatmap(np.log10(MSEsAdagrad), annot=True, ax=ax1, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'MSE (log 10)'},annot_kws={"size": 12})
    cbar = ax1.collections[0].colorbar
    cbar.set_label('MSE (log 10)', fontsize = 15)
    ax1.set_title(f"MSE for Adagrad, {activation.__name__} activation, {initialization} scheme", fontsize = 20)
    ax1.set_ylabel("Learning rate", fontsize = 15)
    ax1.set_xlabel("Lambda", fontsize = 15)
    ax1.add_patch(plt.Rectangle((indexoptimalAdagrad[1], indexoptimalAdagrad[0]), 1, 1, fill=False, edgecolor='orange', lw=3))
    plt.savefig(MSE_PATH + f"/MSE_Adagrad_own_{activation.__name__}_{initialization}_{data}.png")
    plt.show()
    

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsRMSprop)
    sns.heatmap(np.log10(MSEsRMSprop), annot=True, ax=ax2, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'MSE (log 10)'},annot_kws={"size": 12})
    cbar = ax2.collections[0].colorbar
    cbar.set_label('MSE (log 10)', fontsize = 15)
    ax2.set_title(f"MSE for RMSprop, {activation.__name__} activation, {initialization} scheme", fontsize = 20)
    ax2.set_ylabel("Learning rate", fontsize = 15)
    ax2.set_xlabel("Lambda", fontsize = 15)
    ax2.add_patch(plt.Rectangle((indexoptimalRMSprop[1], indexoptimalRMSprop[0]), 1, 1, fill=False, edgecolor='orange', lw=3))
    plt.savefig(MSE_PATH + f"/MSE_RMSprop_own_{activation.__name__}_{initialization}_{data}.png")
    plt.show()
    

    fig3, ax3 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsADAM)
    sns.heatmap(np.log10(MSEsADAM), annot=True, ax=ax3, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'MSE (log 10)'},annot_kws={"size": 12})
    cbar = ax3.collections[0].colorbar
    cbar.set_label('MSE (log 10)', fontsize = 15)
    ax3.set_title(f"MSE for ADAM, {activation.__name__} activation, {initialization} scheme", fontsize = 20)
    ax3.set_ylabel("Learning rate", fontsize = 15)
    ax3.set_xlabel("Lambda", fontsize = 15)
    ax3.add_patch(plt.Rectangle((indexoptimalADAM[1], indexoptimalADAM[0]), 1, 1, fill=False, edgecolor='orange', lw=3))
    plt.savefig(MSE_PATH + f"/MSE_ADAM_own_{activation.__name__}_{initialization}_{data}.png")
    plt.show()
    
    
    # Plot heatmaps with NaN values highlighted (R2 score)
    
    fig5, ax5 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sAdagrad) | (R2sAdagrad < 0)
    sns.heatmap(R2sAdagrad, annot=True, ax=ax5, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'R2'},annot_kws={"size": 12})
    cbar = ax5.collections[0].colorbar
    cbar.set_label('R2', fontsize = 15)
    ax5.set_title(f"R2 for Adagrad, {activation.__name__} activation, {initialization} scheme", fontsize = 20)
    ax5.set_ylabel("Learning rate", fontsize = 15)
    ax5.set_xlabel("Lambda", fontsize = 15)
    ax5.add_patch(plt.Rectangle((indexoptimalAdagrad[1], indexoptimalAdagrad[0]), 1, 1, fill=False, edgecolor='orange', lw=3))
    plt.savefig(R2_PATH + f"/R2_Adagrad_own_{activation.__name__}_{initialization}_{data}.png")
    plt.show()
    
    
    fig6, ax6 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sRMSprop) | (R2sRMSprop < 0)
    sns.heatmap(R2sRMSprop, annot=True, ax=ax6, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'R2'},annot_kws={"size": 12})
    cbar = ax6.collections[0].colorbar
    cbar.set_label('R2', fontsize = 15)
    ax6.set_title(f"R2 for RMSprop, {activation.__name__} activation, {initialization} scheme", fontsize = 20)
    ax6.set_ylabel("Learning rate", fontsize = 15)
    ax6.set_xlabel("Lambda", fontsize = 15)
    ax6.add_patch(plt.Rectangle((indexoptimalRMSprop[1], indexoptimalRMSprop[0]), 1, 1, fill=False, edgecolor='orange', lw=3))
    plt.savefig(R2_PATH + f"/R2_RMSprop_own_{activation.__name__}_{initialization}_{data}.png")
    plt.show()
    

    fig7, ax7 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sADAM) | (R2sADAM < 0)
    sns.heatmap(R2sADAM, annot=True, ax=ax7, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'R2'},annot_kws={"size": 12})
    cbar = ax7.collections[0].colorbar
    cbar.set_label('R2', fontsize = 15)
    ax7.set_title(f"R2 for ADAM, {activation.__name__} activation, {initialization} scheme", fontsize = 20)
    ax7.set_ylabel("Learning rate", fontsize = 15)
    ax7.set_xlabel("Lambda", fontsize = 15)
    ax7.add_patch(plt.Rectangle((indexoptimalADAM[1], indexoptimalADAM[0]), 1, 1, fill=False, edgecolor='orange', lw=3))
    plt.savefig(R2_PATH + f"/R2_ADAM_own_{activation.__name__}_{initialization}_{data}.png")
    plt.show()
    
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sSGD) | (R2sSGD < 0)
    sns.heatmap(R2sSGD, annot=True, ax=ax4, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'R2'},annot_kws={"size": 12})
    cbar = ax4.collections[0].colorbar
    cbar.set_label('R2', fontsize = 15)
    ax4.set_title(f"R2 for SGD, {activation.__name__} activation, {initialization} scheme", fontsize = 20)
    ax4.set_ylabel("Learning rate", fontsize = 15)
    ax4.set_xlabel("Lambda", fontsize = 15)
    ax4.add_patch(plt.Rectangle((indexoptimalSGD[1], indexoptimalSGD[0]), 1, 1, fill=False, edgecolor='orange', lw=3))
    plt.savefig(R2_PATH + f"/R2_SGD_own_{activation.__name__}_{initialization}_{data}.png")
    plt.show()
    # ------------- Storing negative R2 values and respective parameter settings ---------------#
    LOST_PATH = os.path.join(INITIALIZATION_ID, "Negative R2s")
    if not os.path.exists(LOST_PATH):
        os.makedirs(LOST_PATH)
    with open(LOST_PATH + f"/MSE_R2_own_{activation.__name__}_{initialization}_{data}_lost.txt","w") as file:
        file.write(f"Models with a negative R2 score for {activation.__name__}, {initialization} scheme, on the {data} dataset:\n") #Reset the file if the program is re-ran with the same analysis settings
        file.write("SGD:\n")
        for i in lostinfoSGD:
            file.write(f"Learning rate: {i[0]}, Lambda: {i[1]}, R2 score: {i[2]}\n")
        file.write("Adagrad:\n")
        for i in lostinfoAdagrad:
            file.write(f"Learning rate: {i[0]}, Lambda: {i[1]}, R2 score: {i[2]}\n")
        file.write("RMSprop:\n")
        for i in lostinfoRMSprop:
            file.write(f"Learning rate: {i[0]}, Lambda: {i[1]}, R2 score: {i[2]}\n")
        file.write("ADAM:\n")
        for i in lostinfoADAM:
            file.write(f"Learning rate: {i[0]}, Lambda: {i[1]}, R2 score: {i[2]}\n")

    # ------------- Storing the optimal values for future comparison --------------#
    
    TXT_PATH = os.path.join(INITIALIZATION_ID, "Optimal models")
    if not os.path.exists(TXT_PATH):
        os.makedirs(TXT_PATH)
    
    with open(TXT_PATH + f"/MSE_R2_own_{activation.__name__}_{initialization}_{data}.txt","w") as file:
        file.write(f"Best models with each regression method for {activation.__name__}, {initialization} scheme, on the {data} dataset:\n") #Reset the file if the program is re-ran with the same analysis settings
    
    
    
    with open(TXT_PATH + f"/MSE_R2_own_{activation.__name__}_{initialization}_{data}.txt","a") as file:
        file.write(f"SGD: Best MSE = {optimalMSESGD}; Best R2 = {optimalR2SGD}; Best eta = {optimaletaSGD}; Best lambda = {optimalmbdSGD}\n")
        file.write(f"Adagrad: Best MSE = {optimalMSEAdagrad}; Best R2 = {optimalR2Adagrad}; Best eta = {optimaletaAdagrad}; Best lambda = {optimalmbdAdagrad}\n")
        file.write(f"RMSprop: Best MSE = {optimalMSERMSprop}; Best R2 = {optimalR2RMSprop}; Best eta = {optimaletaRMSprop}; Best lambda = {optimalmbdRMSprop}\n")
        file.write(f"ADAM: Best MSE = {optimalMSEADAM}; Best R2 = {optimalR2ADAM}; Best eta = {optimaletaADAM}; Best lambda = {optimalmbdADAM}\n")


elif analysis_type == 1:
    ANALYSIS_ID = os.path.join(DATASET_ID, "Bootstrap")
    if not os.path.exists(ANALYSIS_ID):
        os.makedirs(ANALYSIS_ID)
    
    Method_type = int(input("Which gradient descent method do you wish to use? \n Press 0 for GD \n Press 1 for SGD\n Press 2 for Adagrad\n Press 3 for RMSprop\n Press 4 for ADAM\n"))
    eta = float(input("Select the value of the learning rate:\n "))
    lamb = float(input("Select the value of the regularization parameter:\n "))
    
    if Method_type == 0:
        Method = "GD"
        momentum = float(input("Select the value of the momentum parameter:\n "))
    elif Method_type == 1:
        Method = "SGD"
        momentum = float(input("Select the value of the momentum parameter:\n "))
    elif Method_type == 2:
        Method = "Adagrad"
        momentum = float(input("Select the value of the momentum parameter:\n "))
    elif Method_type == 3:
        Method = "RMSprop"
        momentum = 0
    elif Method_type == 4:
        Method = "Adam"
        momentum = 0
    else:
        sys.exit()
        
    print("Default settings: 10 batches\n 100 epochs\n 1 hidden layer\n 100 hidden neurons")
    bootstrap_type = int(input("Which architectural feature do you wish to vary? \n Press 0 for number of batches \n Press 1 for number of epochs\n Press 2 for number of neurons\n Press 3 for number of layers\n"))
    if bootstrap_type == 0:
        variable = "batches"
    elif bootstrap_type == 1:
        variable = "epochs"
    elif bootstrap_type == 2:
        variable = "neurons"
    elif bootstrap_type == 3:
        variable = "layers"
    else:
        sys.exit()
    
    BOOTSTRAP_ID = os.path.join(ANALYSIS_ID, f"{variable}")
    if not os.path.exists(BOOTSTRAP_ID):
        os.makedirs(BOOTSTRAP_ID)    
    
    with open(BOOTSTRAP_ID+ f"/{activation.__name__}_{initialization}_{variable}_train.txt","w") as file:
        file.write(f"Bias-variance for validation data using {activation.__name__} activation, {initialization}, {Method} with L.R. = {eta} and Reg. = {lamb} on the {data} dataset\n")
    with open(BOOTSTRAP_ID+ f"/{activation.__name__}_{initialization}_{variable}_test.txt","w") as file:
        file.write(f"Bias-variance for validation data using {activation.__name__} activation, {initialization}, {Method} with L.R. = {eta} and Reg. = {lamb} on the {data} dataset\n")
        
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)
        
    """
    Set limits for the variations of batches, epochs, neurons and layers
    """
    if bootstrap_type == 0:
        mini, maxi = 1, 100
    elif bootstrap_type == 1:
        mini, maxi = 1, 200
    elif bootstrap_type == 2:
        mini, maxi = 10, 200
    elif bootstrap_type == 3:
        mini, maxi = 1, 7
    testvar = np.arange(mini, maxi+1, 1) #whatever we're trying to test
        
        
    error_test    = np.zeros(len(testvar))
    bias_test     = np.zeros(len(testvar))
    variance_test = np.zeros(len(testvar))
        
    error_train = np.zeros(len(testvar))
    bias_train = np.zeros(len(testvar))
    variance_train = np.zeros(len(testvar))
        
    for i in tqdm(range(len(testvar))):
            
        #same number of points in the x and y direction
        #creates an array with n_inputs lines and 20 (the number of bootstraps) columns
        z_pred = np.empty((X_test.shape[0], 20))
        z_tilde = np.empty((X_train.shape[0], 20)) #pred is for test data, tilde for train data
            
            
            
        """
        initiates a network whose parameters vary according to what we are
        bootstrapping for
            
        i is varying between 0 and len(testvar)-1 -> testvar[i] is the current
        number of the variable that we are bootstrapping for
        """
        if bootstrap_type == 2:
            sizes = [X.shape[1],testvar[i],1]
        elif bootstrap_type == 3:
            sizes = [X.shape[1]]+[100 for k in range(testvar[i])]+[1]
        else:
            sizes = [X.shape[1],100,1]
                
        MLP = Network(sizes, activation, identity, CostOLS, initialization) 
        for j in range(20):
            
            #make resampled design matrix, targets (scaling is handled by the network fit)
                
            X_, z_ = resample(X_train, z_train)

                
            #for every bootstrap, the network must be reset
            MLP.reset_weights()
            if bootstrap_type == 0:
                MLP.fit(X_, z_, n_batches = testvar[i], n_epochs = 100, eta = eta, lmb = lamb, delta_mom = momentum, method = Method)
            elif bootstrap_type == 1: #disable early stopping when bootstrapping for epochs, otherwise the effect can't be studied
                MLP.fit(X_, z_, n_batches = 10, n_epochs = testvar[i], eta = eta, lmb = lamb, delta_mom = momentum, method = Method, early_stopping = False)
            else:
                MLP.fit(X_, z_, n_batches = 10, n_epochs = 100, eta = eta, lmb = lamb, delta_mom = momentum, method = Method)
                
            # Evaluate the new model on the same testing and training data each time.
            # The j-th column of z_pred corresponds to the j-th bootstrap
            z_pred[:, j]  = MLP.predict(X_, z_, X_test, z_test)[2].flatten()
            z_tilde[:, j] = MLP.predict(X_, z_, X_train, z_train)[2].flatten()
            
            
        error_test[i]    = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias_test[i]     = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance_test[i] = np.mean( np.var(z_pred, axis=1, keepdims=True)                )
        
        error_train[i]    = np.mean( np.mean((z_train - z_tilde)**2, axis=1, keepdims=True) )
        bias_train[i]     = np.mean( (z_train - np.mean(z_tilde, axis=1, keepdims=True))**2 )
        variance_train[i] = np.mean( np.var(z_tilde, axis=1, keepdims=True)                 )
            
        with open(BOOTSTRAP_ID+ f"/{activation.__name__}_{initialization}_{variable}_train.txt","a") as file:
            file.write(f"{testvar[i]} {variable}: MSE {error_train[i]}, Bias {bias_train[i]}, Variance {variance_train[i]}\n")
        with open(BOOTSTRAP_ID+ f"/{activation.__name__}_{initialization}_{variable}_test.txt","a") as file:
            file.write(f"{testvar[i]} {variable}: MSE {error_test[i]}, Bias {bias_test[i]}, Variance {variance_test[i]}\n")
        
    step = int(len(testvar)/5) 
        
    plt.figure(dpi=200)
    plt.plot(testvar, error_train, label='Mean Squared Error', color='red')
    plt.plot(testvar, bias_train, label='Bias', color='blue')
    plt.plot(testvar, variance_train, label='Variance', color='lime')
    plt.xticks(testvar[::step])
    plt.xlabel(f'Number of {variable}', fontsize = 15)
    plt.ylabel('Error (training data)', fontsize = 15)
    plt.title(f'{activation.__name__}, {initialization},{Method} η = {eta} and λ = {lamb}', fontsize = 15)
    plt.legend()
    plt.grid()
    plt.savefig(BOOTSTRAP_ID + f"/Bias_variance_{activation.__name__}_{initialization}_{variable}_train.png")
    plt.show()
        
    plt.figure(dpi=200)
    plt.plot(testvar, error_test, label='Mean Squared Error', color='red')
    plt.plot(testvar, bias_test, label='Bias', color='blue')
    plt.plot(testvar, variance_test, label='Variance', color='lime')
    plt.xticks(testvar[::step])
    plt.xlabel(f'Number of {variable}', fontsize = 15)
    plt.ylabel('Error (validation data)', fontsize = 15)
    plt.title(f'{activation.__name__}, {initialization}, {Method} η = {eta} and λ = {lamb}', fontsize = 15)
    plt.legend()
    plt.grid()
    plt.savefig(BOOTSTRAP_ID + f"/Bias_variance_{activation.__name__}_{initialization}_{variable}_test.png")
    plt.show()
else:
    sys.exit()
#------------------------- Testing score over epochs for different weight initializations (1D data) --------------------#
"""
This analysis is conducted as a small aside to the rest of the plotting, just
to show a comparison between the convergence rate for the 3 used initialization
schemes -> will be relevant for cutting some analysis for the Franke dataset

The analysis is conducted on the optimal settings found for each activation/initialization pair
after the gridsearch. early_stopping is set to false to allow for a proper visualization
of all 100 training epochs
"""
if dataset == 0:
    ANALYSIS_ID = os.path.join(DATASET_ID, "Different initializations")
    if not os.path.exists(ANALYSIS_ID):
        os.makedirs(ANALYSIS_ID)
    
    MLPlist = list()
    
    scorelists = list()
    activations, initializations = [sigmoid, RELU, LRELU], ["RANDN", "HE", "XAVIER"]   
    for i in activations:
        for j in initializations:
            MLPlist.append(Network([1,100,1], i, identity, CostOLS, j))
    optimalparams = [("Adam",1,0.001),("RMSprop",0.001,0),("RMSprop",0.1,0.0001),("Adam",0.1,0.01),("Adam",0.1,0.0),("Adam",0.01,0.0),("Adam",0.1,0.01),("Adam",0.1,0.0001),("Adam",0.01,0.0)] #manually created list with the optimal regression method, eta, lambda that gave the best results
    
    for i,j in zip(MLPlist,optimalparams):
        i.reset_weights()
        scorelists.append(i.fit(X, z, n_batches = 10, n_epochs = 100, eta = j[1], lmb = j[2], delta_mom = 0, method = j[0], early_stopping = True)[2])
    plt.figure()
    plt.yscale('log')
    plt.plot(range(20), scorelists[0][:20], label='N(0,1) weights and biases', color='red')
    plt.plot(range(20), scorelists[1][:20], label='He', color='blue')
    plt.plot(range(20), scorelists[2][:20], label='Xavier', color='lime')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.title('Comparison of initializations for sigmoid activation', fontsize = 10)
    plt.legend()
    plt.grid()
    plt.savefig(ANALYSIS_ID + "/Initialization_comparison_sigmoid.png")
    plt.show()
    
    plt.figure()
    plt.yscale('log')
    plt.plot(range(20), scorelists[3][:20], label='N(0,1) weights and biases', color='red')
    plt.plot(range(20), scorelists[4][:20], label='He', color='blue')
    plt.plot(range(20), scorelists[5][:20], label='Xavier', color='lime')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.title('Comparison of initializations for RELU activation', fontsize = 10)
    plt.legend()
    plt.grid()
    plt.savefig(ANALYSIS_ID + "/Initialization_comparison_RELU.png")
    plt.show()
    
    plt.figure()
    plt.yscale('log')
    plt.plot(range(20), scorelists[6][:20], label='N(0,1) weights and biases', color='red')
    plt.plot(range(20), scorelists[7][:20], label='He', color='blue')
    plt.plot(range(20), scorelists[8][:20], label='Xavier', color='lime')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.title('Comparison of initializations for LRELU activation', fontsize = 10)
    plt.legend()
    plt.grid()
    plt.savefig(ANALYSIS_ID + "/Initialization_comparison_LRELU.png")
    plt.show()
    
"""
To do: test different numbers of hidden neurons with 1 hidden layer, 
using the optimal eta obtained in part a). After conclusions drawn (maybe MSE 
against number os neurons plot), fix number of neurons and tune eta and lambda
    
    
Argue against testing more layers with the Universal Approach Theorem (too many
hyperparameters to keep track of...)


Questions in the project:
    
    1. Why the quadratic cost function? Because it is sensitive to outliers and it
    allows for the implementation of backpropagation, as the cost for the whole
    database (or for a minibatch) is simply the average of the costs for each 
    point
    
    2. The biases are initialized with a small value to ensure all neurons
    have some output for backpropagation in the first training cycle cite{Lecture notes}.
    
    
    3. Activation function for the output layer: for a linear regression task,
    no activation function is needed (or, in other words, the activation function
    f(z) is simply z).
"""    
