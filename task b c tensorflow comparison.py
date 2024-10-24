# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:04:36 2024

@author: User
"""

#-------------------------- Imports -------------------------#

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

import autograd.numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


#----------------------- Scaler class -------------------------#

class Scaler:
    def __init__(self, classification):
        self.classification = classification
        if not self.classification:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
        else:
            self.scaler_X = MinMaxScaler()
            self.scaler_y = OneHotEncoder()
        

    def scaletrain(self, X, y):
        # Scale both features (X) and target (y) - scales differently
        # depending on the type of problem
        
        X_scaled = self.scaler_X.fit_transform(X)
        
        if not self.classification:    
            y_scaled = self.scaler_y.fit_transform(y)
            return X_scaled, y_scaled
        
        elif self.classification=="Binary":
            y_scaled = y
            return X_scaled, y_scaled
        
        elif self.classification == "Multiclass":
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).toarray()
            return X_scaled, y_scaled
        
    def scaletest(self, X, y):
        #Scale testing data in the same way as training data
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)
        return X_scaled, y_scaled
    
    def rescale(self, y_pred):
        # Rescale predictions (y_pred) to the original scale
        if not self.classification:
            return self.scaler_y.inverse_transform(y_pred)
        else:
            pass;
            
#----------------------- Bools for tests to conduct ------------#

want_1D = True

want_franke = True             
#----------------------- Data sets ----------------------------#
if want_1D:
    want_franke = False
    np.random.seed(2024)

    n = 1000
    x = 2*np.random.rand(n,1)
    y = 2 + x + 2*x**2 + np.random.randn(n,1)

    scaler1 = Scaler(classification = False)

    # Creating Design Matrix
    X = np.c_[x, x**2] # No Intecept given Scaling-process


    X_scaled, y_scaled = scaler1.scaletrain(X, y)

    MSEvalues=[]
    
    eta_vals = np.logspace(-4, 1, 6)
    lmbd_vals = np.logspace(-5, 1, 7)

#----------------------- Creating the neural network ------------------------#


# Lecture notes from week 43, improved with GPT


def NN_model(inputsize, n_layers, n_neuron, optimizer, eta, lamda):
    model = Sequential()

    # Add an explicit Input layer
    model.add(Input(shape=(inputsize,)))
    
    for i in range(n_layers):
        model.add(Dense(n_neuron, activation='sigmoid', kernel_regularizer=regularizers.l2(lamda)))
    
    model.add(Dense(1))  # 1 output, no activation for linear regression

    # Choose the optimizer
    if optimizer == "ADAGRAD":
        optimizerr = optimizers.Adagrad(learning_rate=eta, initial_accumulator_value=0, epsilon=1e-08)
    elif optimizer == "ADAM":   
        optimizerr = optimizers.Adam(learning_rate=eta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == "RMSPROP":   
        optimizerr = optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-08)

    # Compile the model with mean_squared_error and custom R² score
    model.compile(loss='mean_squared_error', optimizer=optimizerr, metrics=['mse', 'R2Score'])

    return model




if want_1D:
    # grid search
    MSEsAdagrad = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    MSEsRMSprop = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    MSEsADAM = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    
    R2sAdagrad = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    R2sRMSprop = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    R2sADAM = np.zeros( ( len(eta_vals), len(lmbd_vals) ) )
    
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            print(f"Eta: {eta}, lambda: {lmbd}")
            ModelADAM = NN_model(2, 3, 100, 'ADAM', eta, lmbd)
            ModelADAM.fit(X_scaled, y_scaled, epochs=100, batch_size=100, verbose=0)
            y_predADAM = ModelADAM.predict(X_scaled)
            y_predADAM = scaler1.rescale(y_predADAM)
            MSEsADAM[i][j] = mean_squared_error(y, y_predADAM)
            R2sADAM[i][j] = r2_score(y, y_predADAM)
            
            ModelAdagrad = NN_model(2, 3, 100, 'ADAGRAD', eta, lmbd)
            ModelAdagrad.fit(X_scaled, y_scaled, epochs=100, batch_size=100, verbose=0)
            y_predAdagrad = ModelAdagrad.predict(X_scaled)
            y_predAdagrad = scaler1.rescale(y_predAdagrad)
            MSEsAdagrad[i][j] = mean_squared_error(y, y_predAdagrad)
            R2sAdagrad[i][j] = r2_score(y, y_predAdagrad)
            
            ModelRMSprop = NN_model(2, 3, 100, 'RMSPROP', eta, lmbd)
            ModelRMSprop.fit(X_scaled, y_scaled, epochs=100, batch_size=100, verbose=0)
            y_predRMSprop = ModelRMSprop.predict(X_scaled)
            y_predRMSprop = scaler1.rescale(y_predAdagrad)
            MSEsRMSprop[i][j] = mean_squared_error(y, y_predAdagrad)
            R2sRMSprop[i][j] = r2_score(y, y_predAdagrad)
            
    # Assuming eta_vals and lmbd_vals are logarithmic ranges like np.logspace(-6, 0, 7) or similar
    eta_log_vals = np.log10(eta_vals)
    lmbd_log_vals = np.log10(lmbd_vals)
      
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsAdagrad)
    sns.heatmap(MSEsAdagrad, annot=True, ax=ax1, cmap="viridis", mask=mask, xticklabels=range(-5,2), yticklabels=range(-4,2), cbar_kws={'label': 'MSE'})
    ax1.set_title("MSE for Adagrad, 10 batches, 100 epochs")
    ax1.set_ylabel("log10 (Learning rate)")
    ax1.set_xlabel("log10(Lambda)")
    plt.savefig("MSE Adagrad TF sigmoid.png")
    plt.show()
    

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsRMSprop)
    sns.heatmap(MSEsRMSprop, annot=True, ax=ax2, cmap="viridis", mask=mask, xticklabels=range(-5,2), yticklabels=range(-4,2), cbar_kws={'label': 'MSE'})
    ax2.set_title("MSE for RMSprop, 10 batches, 100 epochs")
    ax2.set_ylabel("log10 (Learning rate)")
    ax2.set_xlabel("log10(Lambda)")
    plt.savefig("MSE RMSprop TF sigmoid.png")
    plt.show()
    

    fig3, ax3 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsADAM)
    sns.heatmap(MSEsADAM, annot=True, ax=ax3, cmap="viridis", mask=mask, xticklabels=range(-5,2), yticklabels=range(-4,2), cbar_kws={'label': 'MSE'})
    ax3.set_title("MSE for ADAM, 10 batches, 100 epochs")
    ax3.set_ylabel("log10 (Learning rate)")
    ax3.set_xlabel("log10 (Lambda)")
    plt.savefig("MSE ADAM TF sigmoid.png")
    plt.show()
    
    
    # Plot heatmaps with NaN values highlighted (R2 score)
      
    fig5, ax5 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sAdagrad)
    sns.heatmap(R2sAdagrad, annot=True, ax=ax5, cmap="viridis", mask=mask, xticklabels=range(-5,2), yticklabels=range(-4,2), cbar_kws={'label': 'R2'})
    ax5.set_title("R2 for Adagrad, 10 batches, 100 epochs")
    ax5.set_ylabel("log10 (Learning rate)")
    ax5.set_xlabel("log10(Lambda)")
    plt.savefig("R2 Adagrad TF sigmoid.png")
    plt.show()
    
    
    fig6, ax6 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sRMSprop)
    sns.heatmap(R2sRMSprop, annot=True, ax=ax6, cmap="viridis", mask=mask, xticklabels=range(-5,2), yticklabels=range(-4,2), cbar_kws={'label': 'R2'})
    ax6.set_title("R2 for RMSprop, 10 batches, 100 epochs")
    ax6.set_ylabel("log10 (Learning rate)")
    ax6.set_xlabel("log10(Lambda)")
    plt.savefig("R2 RMSprop TF sigmoid.png")
    plt.show()
    

    fig7, ax7 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sADAM)
    sns.heatmap(R2sADAM, annot=True, ax=ax7, cmap="viridis", mask=mask, xticklabels=range(-5,2), yticklabels=range(-4,2), cbar_kws={'label': 'R2'})
    ax7.set_title("R2 for ADAM, 10 batches, 100 epochs")
    ax7.set_ylabel("log10 (Learning rate)")
    ax7.set_xlabel("log10 (Lambda)")
    plt.savefig("R2 ADAM TF sigmoid.png")
    plt.show()
    
    # ------------- Storing the optimal values for future comparison --------------#
    """
    Gonna send them to a .txt file, then will format a table to compare optimal MSE
    eta and lambda for my implementation. Will do the same for Tensorflow. Will then
    use the best parameter and method settings to fit the terrain data (hopefully it doesn't
    take a billion years).
    
    Will also redo this for RELU and LRELU. A study of different bias initialization is probably
    better suited for the classification question, since that is the one where we encounter vanishing
    gradients.
    """
    with open("MSE_R2_TF_sig.txt","w") as file:
        file.write(" ") #Reset the file if the program is re-ran
    
    # Optimal MSE and corresponding indices
    optimalMSEAdagrad, indexoptimalAdagrad = np.min(MSEsAdagrad), np.unravel_index(np.argmin(MSEsAdagrad), MSEsAdagrad.shape)
    optimalMSERMSprop, indexoptimalRMSprop = np.min(MSEsRMSprop), np.unravel_index(np.argmin(MSEsRMSprop), MSEsRMSprop.shape)
    optimalMSEADAM, indexoptimalADAM = np.min(MSEsADAM), np.unravel_index(np.argmin(MSEsADAM), MSEsADAM.shape)
    
    # "Optimal" R2 (in principle, the highest R2 should correspond to the lowest MSE,
    # either way, MSE takes a higher priority in our analysis) 
    optimalR2Adagrad = R2sAdagrad[indexoptimalAdagrad[0]][indexoptimalAdagrad[1]]
    optimalR2RMSprop = R2sRMSprop[indexoptimalRMSprop[0]][indexoptimalRMSprop[1]]
    optimalR2ADAM = R2sADAM[indexoptimalADAM[0]][indexoptimalADAM[1]]
    
    optimaletaAdagrad, optimalmbdAdagrad = eta_vals[indexoptimalAdagrad[0]], lmbd_vals[indexoptimalAdagrad[1]]
    optimaletaRMSprop, optimalmbdRMSprop = eta_vals[indexoptimalRMSprop[0]], lmbd_vals[indexoptimalRMSprop[1]]
    optimaletaADAM, optimalmbdADAM = eta_vals[indexoptimalADAM[0]], lmbd_vals[indexoptimalADAM[1]]
    
    with open("MSE_R2_TF_sig.txt","a") as file:
        file.write(f"Adagrad: Best MSE = {optimalMSEAdagrad}; Best R2 = {optimalR2Adagrad}; Best eta = {optimaletaAdagrad}; Best lambda = {optimalmbdAdagrad}")
        file.write(f"RMSprop: Best MSE = {optimalMSERMSprop}; Best R2 = {optimalR2RMSprop}; Best eta = {optimaletaRMSprop}; Best lambda = {optimalmbdRMSprop}")
        file.write(f"ADAM: Best MSE = {optimalMSEADAM}; Best R2 = {optimalR2ADAM}; Best eta = {optimaletaADAM}; Best lambda = {optimalmbdADAM}")

"""
if want_franke:
    
    want_neurons = False
    want_gridsearch = False #Safeguards in case I forget to manually switch them off
    #-------------- Taken from the task a) code from last project
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)
    deg_max = 6  # up to degree 5
    
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    X = np.vstack((x_flat,y_flat)).T

    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)

        
    z_train = z_train.reshape(-1,1)
    z_test = z_test.reshape(-1,1)  #they are only reshaped for the purposes of train test split

    for deg in range(1,deg_max):
        Phi_train = Design_Matrix_2D(deg, X_train)
        Phi_test  = Design_Matrix_2D(deg, X_test) 
        
        num_feats = int((deg + 1) *(deg + 2) / 2 - 1)
        
        MLP = Network([num_feats,100,1], sigmoid, identity, CostOLS) # Our model now has 1595 features
        MLP.set_classification()
        MLP.reset_weights()
        MLP.fit(Phi_train, z_train, n_batches = 10, n_epochs = 100, eta = 0.0001, lmb = 0.01, delta_mom = 0, method = 'RMSprop', scale_bool = True, tol = 1e-6)
    
        z_pred = MLP.predict(Phi_train, z_train, Phi_test, z_test, scale_bool = True)
    
        print(f"MSE for Franke's Function with degree {deg}: {mean_squared_error(z_test,z_pred)}")

"""