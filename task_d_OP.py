# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:41:00 2024

@author: User
"""

#---------------------imports-------------------------------#
import autograd.numpy as np
import random
from autograd import grad, elementwise_grad

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, log_loss
import sklearn.datasets

from tqdm import tqdm


import warnings
warnings.simplefilter("error")
random.seed(2023)
#---------------- Cost, activation functions (Lecture notes) -------------#

"""
def CostOLS(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func
"""


def CostLogReg(target):

    def func(X):
        
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):
    
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func


"""
These are defined by an X of any shape. elementwise_grad generalizes grad for 
vector/matrix output functions such as these
"""

def identity(X):
    return X


def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


#-------------------Automatic differentiation------------#
def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return np.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)

        return func

    else:
        return elementwise_grad(func)



#----------------------------- Testing functions for producing data ----------------------#

def Design_Matrix_2D(deg, X):
    # The number of polynomial terms for two variables (x, y) up to degree d is (d+1)(d+2)/2  
    # Minus 1 from dropping intercept-column
    num_terms = int((deg + 1) * (deg + 2) / 2 - 1)

    Phi = np.zeros((X.shape[0], num_terms))
    # PS: not include intercept in design matrix, will scale (centered values)
    col = 0
    for dx in range(1,deg + 1):
        for dy in range(dx + 1):
            # X[:,0] = x-values
            # X[:,1] = y-values
            Phi[:,col] = (X[:,0] ** (dx - dy)) * (X[:,1] ** dy)
            col += 1
    return Phi



#------------------- Class for scaling and rescaling using StandardScaler -------#

class Scaler:
    def __init__(self, classification):
        self.classification = classification
        if not self.classification:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
        else:
            self.scaler_X = MinMaxScaler()
            self.scaler_y = OneHotEncoder(sparse_output=False)
        

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
        if self.classification == "Binary":
            pass
        else:
            return self.scaler_y.inverse_transform(y_pred)
            


#Class for neural networks

class Network(object):

    def __init__(self, sizes, hiddenact, outputact, costfunc, seed=2024): #taken from Michael Nielsen
        """
        sizes is a list with the number of neurons per each hidden layer
        X_data and y_data are the design matrix and target variables for the
        training data (they are already included in the class implementation)
        
        actfunc sets the activation function for the hidden layers
        """
        
        self.seed = seed #seed for reproducibility
        np.random.seed(self.seed)
        
        self.hiddenact = hiddenact
        self.outputact = outputact
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1, y) * 0.01  for y in sizes[1:]] 
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        
        
        # ------ LECTURE NOTES FOR WEEK 42 ------ #
        self.a_matrices = list() # list to store all the activations, layer by layer
        self.z_matrices = list() # list to store all the z vectors, layer by layer
        self.costfunc = costfunc
        self.classification = None
    
    
    
    def feedforward(self, X): #taken from Michael Nielsen
        """
        The feedforward method will have as an output the lists of activations and z's 
        (allowing for backpropagation)
        """
        # Resetting a and z matrices
        self.a_matrices = list()
        self.z_matrices = list()
        
        # Make X into a matrix if it is just a vector (Lecture notes)
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))
    
        a = X #activations for input layer are the inputs themselves
        self.a_matrices.append(a) 
        
        self.z_matrices.append(a) 
        
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = a @ w + b
            self.z_matrices.append(z)
            
            a = self.hiddenact(z)
            self.a_matrices.append(a)
            
        #The output has a different activation function
        z = a @ self.weights[-1] + self.biases[-1]
        self.z_matrices.append(z)
        
        a=self.outputact(z)
        self.a_matrices.append(a)
        
        #print(a.shape)
        return a
    
    
    
    def backprop(self, X, y, lmb): #taken from Michael Nielsen
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #derivatives of activation functions (LECTURE NOTES)
        out_derivative = derivate(self.outputact)
        hidden_derivative = derivate(self.hiddenact)
        
        #Delta and gradients for output layer
        """
        self.z_matrices[-1] are the output zs. self.a_matrices[-1] are the output as
        
        a_matrices[i] and z_matrices[i] have shapes (n_inputs, n_nodes) for a given layer i
        
        therefore, delta_matrix here will have shape (n_inputs, n_output_nodes)
        """
        
        for i in range(len(self.weights)-1, -1, -1):
            # delta terms for output
            if i == len(self.weights) - 1:
                # for multi-class classification
                if (self.outputact.__name__ == "softmax"):
                    delta_matrix = self.a_matrices[i + 1] - y
                # for single class classification
                else:
                    cost_func_derivative = grad(self.costfunc(y))
                    delta_matrix = out_derivative(self.z_matrices[i + 1] ) * cost_func_derivative(self.a_matrices[i + 1])

            # delta terms for hidden layer
            else:
                delta_matrix = (self.weights[i + 1] @ delta_matrix.T).T * hidden_derivative(self.z_matrices[i + 1])

            # calculate gradient
            gradient_weights = self.a_matrices[i].T @ delta_matrix
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

            # regularization term
            gradient_weights += self.weights[i] * lmb
            
            nabla_b[i] = gradient_bias
            nabla_w[i] = gradient_weights
        
        return (nabla_b, nabla_w)
    
    
    
    #-------------- PARTLY TAKEN FROM LECTURE NOTES FROM WEEK 42 ------------------#
    
    def fit(self, X, y, n_batches, n_epochs, eta, lmb, delta_mom, method, scale_bool, tol, threshold = 0.5, t1=1, X_val = None, y_val = None, SGD_bool = True):
     """
     performs a fit of the network using a given gradient descent method (specified
     by "method")
     select eta and lambda values, number of batches, number of epochs and if you
     want scaling, momentum or not (scale_bool, momentum)
     
     
     for non-stochastic methods, n_epochs simply becomes the number of iterations
     """
     if scale_bool:
         X_scaled, y_scaled, scaler = self.scaletraining(X, y)
     else:
         X_scaled, y_scaled = X, y
         
     # setup 
     if self.seed is not None:
         np.random.seed(self.seed)

     val_set = False
     if X_val is not None and y_val is not None:
         val_set = True
     
     if method == "GD":
         self.GD(X, y, eta, lmb, delta_mom, scale_bool, n_epochs, tol)
     
     elif method == "SGD":
         self.SGD(X, y, n_epochs, n_batches, tol, eta, lmb, delta_mom, scale_bool, t1)
         
     elif method == "Adagrad":
         self.Adagrad(X, y, n_epochs, eta, lmb, n_batches, tol, delta_mom, SGD_bool, scale_bool)
         
     elif method == "RMSprop":
         self.RMSprop(X, y, n_epochs, eta, lmb, n_batches, tol, scale_bool)
         
     elif method == "Adam":
         self.Adam(X, y, n_epochs, eta, lmb, n_batches, tol, scale_bool)
         
     cost_function_train = self.costfunc(y)    
     # Performs a prediction with the updated parameters
     #and returns the value of the cost function
     
     
     if not self.classification:
         y_pred=self.feedforward(X_scaled)
         
         if scale_bool:
             y_pred = scaler.rescale(y_pred)
             
         score = cost_function_train(y_pred)
     
         return score
  
  
     elif self.classification == "Multiclass":
         score = self.accuracy(X_scaled, y)
         return score
     

     elif self.classification == "Binary": 
         score0 = self.accuracy(X_scaled, y)[0]
         score1 = self.accuracy(X_scaled, y)[1]
         score2 = self.accuracy(X_scaled, y)[2]
         score3 = self.accuracy(X_scaled, y)[3]
         score4 = self.accuracy(X_scaled, y)[4]
         predictions = self.accuracy(X_scaled, y)[5]

         return (score0, score1, score2, score3, score4, predictions)
    
    
    
    #-------------- LECTURE NOTES FROM WEEK 42-------------#
    
    def reset_weights(self):
        """
        Description:
        ------------
            Resets/Reinitializes the weights in order to train the network for a new problem.

        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = [np.random.randn(x, y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(1, y) * 0.01  for y in self.sizes[1:]] 
    
    
    
    def split_mini_batches (self, nbatches, X, y):
        
        """
        create a list with n minibatches that the data is split into 
        Data is shuffled before making the batches
        """
        
        """
        M = X.shape[0] // nbatches #size of minibatches
        batches=[]
        
        for i in range(nbatches):
            X_batch, y_batch = resample(X, y, replace= True, n_samples=M)
            batches.append((X_batch, y_batch))
        return batches, X, y
        """
        data_size = X.shape[0]
        indices = np.arange(data_size)  # Create an array of indices for shuffling

        # Shuffle the indices without replacement
        np.random.shuffle(indices)

        # Shuffle both input data and target data using the shuffled indices
        shuffled_inputs = X[indices]
        shuffled_targets = y[indices]

        # Split the shuffled data into mini-batches
        M = data_size // nbatches  # Size of each mini-batch
        minibatches = []

        for i in range(nbatches):
            X_batch = shuffled_inputs[i*M:(i+1)*M]
            y_batch = shuffled_targets[i*M:(i+1)*M]
            minibatches.append((X_batch, y_batch))

        return minibatches, shuffled_inputs, shuffled_targets
    
    
    
    def scaletraining (self, x, y):
        """
        What the scaler instance does depends on whether this is a classification
        problem or not
        
        trains the scaler on x,y, transforms them as well and then returns
        
        x_scaled, y_scaled and the trained scaler

        """
        
        # Scale values
        scaler = Scaler(self.classification) # Create a scaler instance
        
        """
        if self.classification is False, creates Standard scalers for X and y
        
        If self.classification is Binary, creates MinMax scaler for X, does nothing
        to y
        
        If self.classification is Multiclass, creates MinMax scaler for X, one hot
        encoder for y
        """

        # Scale the input data and target
        x_scaled, y_scaled = scaler.scaletrain(x, y)
        
        return x_scaled, y_scaled, scaler #save the scaler to invert
    
    
    
    # ------------ From lecture notes ---------- #
    def accuracy(self, X, y, threshold = 0.5):
        
        y_pred = self.feedforward(X)
        
        
        if self.classification == "Multiclass":
            predictions = np.argmax(y_pred, axis=1)  # For multi-class classification
            return np.mean(predictions == np.argmax(y, axis=1))
        
        if self.classification == "Binary":
            predictions = np.where(y_pred > threshold, 1, 0)
            
            #Testing for correlations (true/false positive/negatives)
            
            #Calculate correlations:
            TP = np.sum((predictions == 1) & (y == 1))/len(y)  # True Positives
            TN = np.sum((predictions == 0) & (y == 0))/len(y)  # True Negatives
            FP = np.sum((predictions == 1) & (y == 0))/len(y)  # False Positives
            FN = np.sum((predictions == 0) & (y == 1))/len(y)  # False Negatives
            accuracy = np.mean(predictions == y)
            
            return (accuracy, TP, TN, FP, FN, predictions)
        
        
    
    def set_classification(self):
        """
        Description:
        ------------
            Decides if FFNN acts as classifier (True) og regressor (False),
            sets self.classification during init()
        """
        self.classification = False
        
        if (self.costfunc.__name__ == "CostLogReg"
            or self.costfunc.__name__ == "CostCrossEntropy") and (self.outputact.__name__ != "softmax"):
            
            self.classification = "Binary"
            
        elif (self.costfunc.__name__ == "CostLogReg"
            or self.costfunc.__name__ == "CostCrossEntropy") and (self.outputact.__name__  == "softmax"):
            
            self.classification = "Multiclass"
    
    
    
    #------------------- GRADIENT DESCENT METHODS ---------------#
    
    # Implemented largely with help from chatGPT - check the conversation logs
    #for more detailed info
    
    def GD (self, X, y, eta, lmb, delta_mom, scale_bool, Niter, tol, threshold = 0.5):

        # Establishing the cost function to evaluate the loss at each epoch
        cost_function = self.costfunc(y)   
        
        # Scaling the data if needed
        if scale_bool:
            x_scaled, y_scaled, scaler = self.scaletraining(X, y)
        else:
            x_scaled, y_scaled = X, y
        
        if not self.classification:
            y_pred = self.feedforward(x_scaled)
            
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
                
            score = cost_function(y_pred)
            
        else: 
            ypred=np.where(self.feedforward(x_scaled) > threshold, 1, 0)
            score = self.accuracy(x_scaled, y)
        
        #diff = tol + 1
        iter = 0
        
        change_weights = [np.zeros(i.shape) for i in self.weights]
        change_biases = [np.zeros(i.shape) for i in self.biases]
        if delta_mom != 0:
            while iter < Niter:
                iter += 1
                self.feedforward(x_scaled)
                gradients = self.backprop(x_scaled, y_scaled, lmb)
                                 
                change_biases = [-eta * gb + delta_mom * cb for gb,cb in zip(gradients[0],change_biases)]
                change_weights = [-eta * gw + delta_mom * cw for gw,cw in zip(gradients[1],change_weights)]
            
                self.weights = [w + cw for w,cw in zip(self.weights, change_weights)]                                            # make change
                self.biases = [b + cb for b,cb in zip(self.biases, change_biases)]       
                
                # Check convergence after each iteration                           
                if not self.classification:
                    y_pred = self.feedforward(x_scaled)
                    
                    if scale_bool:
                        y_pred = scaler.rescale(y_pred)
                        
                    newscore = cost_function(y_pred)
                    
                else: 
                    newscore = self.accuracy(x_scaled, y)
                    
                if not self.classification and abs(score-newscore)<=tol:
                    score = newscore
                    print(f"Convergence reached after {iter} iterations.")
                    break;
                score = newscore    
        else:
            while iter < Niter:
                # Will be plain OLS if lmb = 0
                iter += 1
                self.feedforward(x_scaled)
                gradients = self.backprop(X, y, lmb)
                
                change_weights = [-eta * i for i in gradients[1]]
                change_biases = [-eta * i for i in gradients[0]]
                
                self.weights = [w + cw for w,cw in zip(self.weights, change_weights)]
                self.biases = [b + cb for b,cb in zip(self.biases, change_biases)]
                
                # Check convergence after each iteration                           
                if not self.classification:
                    y_pred = self.feedforward(x_scaled)
                    
                    if scale_bool:
                        y_pred = scaler.rescale(y_pred)
                        
                    newscore = cost_function(y_pred)
                    
                else: 
                    newscore = self.accuracy(x_scaled, y)
                    
                if not self.classification and abs(score-newscore)<=tol:
                    score = newscore
                    print(f"Convergence reached after {iter} iterations.")
                    break;
                score = newscore     
    
    
    
    def SGD(self, X, y, n_epochs, n_batches, tol, eta0, lmb, delta_mom, scale_bool, t1, threshold = 0.5):
        
        # Establishing the cost function to evaluate the loss at each epoch
        cost_function = self.costfunc(y)   
        
        # Scaling the data if needed
        if scale_bool:
            x_scaled, y_scaled, scaler = self.scaletraining(X, y)
        else:
            x_scaled, y_scaled = X, y
        
        if not self.classification:
            y_pred = self.feedforward(x_scaled)
            
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
                
            score = cost_function(y_pred)
            
        else: 
            ypred=np.where(self.feedforward(x_scaled) > threshold, 1, 0)
            score = self.accuracy(x_scaled, y)

        n_epochs = int(n_epochs)
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches

        minibatches, x_scaled, y_scaled = self.split_mini_batches(n_batches, x_scaled, y_scaled) # saves the batches and the reshuffled X and y

        #t0 = eta0 * t1  # No longer rbitrary t0, earlier it was set to 1
        """
        The old implementation of time decay got stuck with an MSE
        of 9.95 for the 1D function, regardless of eta or lambda - probably
        because the learning rate decayed so fast that the different initial
        values ended up becoming irrelevant, or because the optimization became
        trapped in a lower minimum.
        """
        # Learning rate decay function (Inverse time decay)
        def time_decay_eta(eta0, t):
            return eta0 / (1 + t/t1)

        for epoch in range(n_epochs):
            #print(epoch, score)
            change_weights = [np.zeros(i.shape) for i in self.weights]
            change_biases = [np.zeros(i.shape) for i in self.biases]
            for i in range(n_batches):
                # Learning rate schedule based on time decay
                eta = time_decay_eta(eta0, epoch * n_batches + i)

                # Choose a minibatch and calculate gradients
                if i == n_batches - 1:
                    X_batch = x_scaled[i * M:, :]
                    y_batch = y_scaled[i * M:, :]
                else:
                    X_batch = minibatches[i][0]
                    y_batch = minibatches[i][1]

                # Forward pass
                self.feedforward(X_batch)

                # Backward pass (compute gradients)
                gradients = self.backprop(X_batch, y_batch, lmb)

                # Momentum update for weights and biases
                change_biases = [delta_mom * cb + (-eta * gb) for gb, cb in zip(gradients[0], change_biases)]
                change_weights = [delta_mom * cw + (-eta * gw) for gw, cw in zip(gradients[1], change_weights)]

                # Update weights and biases
                self.weights = [w + cw for w, cw in zip(self.weights, change_weights)]
                self.biases = [b + cb for b, cb in zip(self.biases, change_biases)]

            # Check convergence after all minibatches for the epoch
                        
            if not self.classification:
                y_pred = self.feedforward(x_scaled)
                
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                    
                newscore = cost_function(y_pred)
                
            else: 
                newscore = self.accuracy(x_scaled, y)
                
            if not self.classification and abs(score-newscore)<=tol:
                score = newscore
                print(f"Convergence reached after {epoch + 1} epochs.")
                break;
            score = newscore
    
    

    def Adagrad(self, X, y, n_epochs, eta, lmb, n_batches, tol, delta_mom, SGD_bool, scale_bool, threshold = 0.5):
        """
        mom_bool and sgd_bool define if you want to use momentum/SGD respectively
        """
        # Establishing the cost function to evaluate the loss at each epoch
        cost_function = self.costfunc(y)   
        
        # Scaling the data if needed
        if scale_bool:
            x_scaled, y_scaled, scaler = self.scaletraining(X, y)
        else:
            x_scaled, y_scaled = X, y
        
        if not self.classification:
            y_pred = self.feedforward(x_scaled)
            
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
                
            score = cost_function(y_pred)
            
        else: 
            ypred=np.where(self.feedforward(x_scaled) > threshold, 1, 0)
            score = self.accuracy(x_scaled, y)
        
        # AdaGrad parameter to avoid possible division by zero
        delta = 1e-8

        # ----------------- SGD - parameters ---------------
        if SGD_bool:
            n_batches = int(n_batches)
            minibatches, x_scaled, y_scaled = self.split_mini_batches(n_batches, x_scaled, y_scaled) # saves the batches and the reshuffled X and y
        else:
            n_batches = 1
        M = X.shape[0] // n_batches
    # ---------------------------------------------------
    
    # Initialize accumulated squared gradients and momentum (if applicable)
        for epoch in range(n_epochs):
            Giter_biases = [np.zeros(b.shape) for b in self.biases]  # Accumulated gradient for biases
            Giter_weights = [np.zeros(w.shape) for w in self.weights]  # Accumulated gradient for weights
        
            change_weights = [np.zeros(w.shape) for w in self.weights]
            change_biases = [np.zeros(b.shape) for b in self.biases]
        
            for i in range(n_batches):
                # Process each minibatch
                if i == n_batches - 1:
                    X_batch = x_scaled[i * M:, :]
                    y_batch = y_scaled[i * M:, :]
                else:
                    X_batch = minibatches[i][0]
                    y_batch = minibatches[i][1]

                # Perform feedforward and backpropagation
                self.feedforward(X_batch)
                gradients = self.backprop(X_batch, y_batch, lmb)

                # Update accumulated gradients (Adagrad update rule)
                Giter_biases = [Gb + g ** 2 for Gb, g in zip(Giter_biases, gradients[0])]
                Giter_weights = [Gw + g ** 2 for Gw, g in zip(Giter_weights, gradients[1])]

                # Compute parameter changes
                change_biases = [eta * g / (np.sqrt(Gb) + delta) for g, Gb in zip(gradients[0], Giter_biases)]
                change_weights = [eta * g / (np.sqrt(Gw) + delta) for g, Gw in zip(gradients[1], Giter_weights)]

                # Apply momentum if enabled
                if delta_mom != 0:
                    change_biases = [cb + delta_mom * cb for cb in change_biases]
                    change_weights = [cw + delta_mom * cw for cw in change_weights]

                # Update weights and biases
                self.biases = [b - cb for b, cb in zip(self.biases, change_biases)]
                self.weights = [w - cw for w, cw in zip(self.weights, change_weights)]
        
            # Check convergence after all minibatches for the epoch
            # Check convergence after all minibatches for the epoch
            
            if not self.classification:
                y_pred = self.feedforward(x_scaled)
                
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                    
                newscore = cost_function(y_pred)
                
            else: 
                newscore = self.accuracy(x_scaled, y)
                
            if not self.classification and abs(score-newscore)<=tol:
                score = newscore
                print(f"Convergence reached after {epoch + 1} epochs.")
                break;

        print("Training complete.")
    
    
    
    def RMSprop(self, X, y, n_epochs, eta, lmb, n_batches, tol, scale_bool, threshold = 0.5):
        """
        RMSprop implementation with a convergence check after each epoch.
        """
        # Establishing the cost function to evaluate the loss at each epoch
        
        cost_function = self.costfunc(y)   
        
        # Scaling the data if needed
        if scale_bool:
            x_scaled, y_scaled, scaler = self.scaletraining(X, y)
        else:
            x_scaled, y_scaled = X, y
        
        if not self.classification:
            y_pred = self.feedforward(x_scaled)
            
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
                
            score = cost_function(y_pred)
            
        else: 
            score = self.accuracy(x_scaled, y)
        
        # Value for parameter rho
        rho = 0.99
        # RMSprop parameter to avoid possible division by zero
        delta = 1e-8

        # ----------------- SGD - parameters ---------------
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches
        minibatches, x_scaled, y_scaled = self.split_mini_batches(n_batches, x_scaled, y_scaled) # saves the batches and the reshuffled X and y
        # ---------------------------------------------------

        # Initialize accumulated squared gradients
        for epoch in range(n_epochs):
            Giter_biases = [np.zeros(b.shape) for b in self.biases]  # Accumulated gradient for biases
            Giter_weights = [np.zeros(w.shape) for w in self.weights]  # Accumulated gradient for weights
        
            change_weights = [np.zeros(w.shape) for w in self.weights]
            change_biases = [np.zeros(b.shape) for b in self.biases]
        
            for i in range(n_batches):
                # Process each minibatch
                if i == n_batches - 1:
                    X_batch = x_scaled[i * M:, :]
                    y_batch = y_scaled[i * M:, :]
                else:
                    X_batch = minibatches[i][0]
                    y_batch = minibatches[i][1]

                # Perform feedforward and backpropagation
                self.feedforward(X_batch)
                gradients = self.backprop(X_batch, y_batch, lmb)

                # Update accumulated gradients (RMSprop update rule)
                Giter_biases = [rho * Gb + (1 - rho) * g ** 2 for Gb, g in zip(Giter_biases, gradients[0])]
                Giter_weights = [rho * Gw + (1 - rho) * g ** 2 for Gw, g in zip(Giter_weights, gradients[1])]

                # Compute parameter changes
                change_biases = [eta * g / (np.sqrt(Gb) + delta) for g, Gb in zip(gradients[0], Giter_biases)]
                change_weights = [eta * g / (np.sqrt(Gw) + delta) for g, Gw in zip(gradients[1], Giter_weights)]

                # Update weights and biases
                self.biases = [b - cb for b, cb in zip(self.biases, change_biases)]
                self.weights = [w - cw for w, cw in zip(self.weights, change_weights)]

            # Check convergence after all minibatches for the epoch
            
            if not self.classification:
                y_pred = self.feedforward(x_scaled)
                
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                    
                newscore = cost_function(y_pred)
                
            else: 
                newscore = self.accuracy(x_scaled, y)
                
            if not self.classification and abs(score-newscore)<=tol:
                score = newscore
                print(f"Convergence reached after {epoch + 1} epochs.")
                break;
            score = newscore    
    
    
    
    def Adam(self, X, y, n_epochs, eta, lmb, n_batches, tol, scale_bool, threshold = 0.5):

        # Establishing the cost function to evaluate the loss at each epoch
        
        cost_function = self.costfunc(y)   
        
        # Scaling the data if needed
        if scale_bool:
            x_scaled, y_scaled, scaler = self.scaletraining(X, y)
        else:
            x_scaled, y_scaled = X, y
        
        if not self.classification:
            y_pred = self.feedforward(x_scaled)
            
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
                
            score = cost_function(y_pred)
            
        else: 
            ypred=np.where(self.feedforward(x_scaled) > threshold, 1, 0)
            score = self.accuracy(x_scaled, y)
        
        # Value for parameters rho1 and rho2, see https://arxiv.org/abs/1412.6980
        rho1 = 0.9
        rho2 = 0.999
        # AdaGrad parameter to avoid possible division by zero
        delta  = 1e-8   
        
        # ----------------- SGD - parameters ---------------
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches
        minibatches, x_scaled, y_scaled = self.split_mini_batches(n_batches, x_scaled, y_scaled) # saves the batches and the reshuffled X and y
        # -------------------------------------------------
        
        s_biases = [np.zeros(i.shape) for i in self.biases]
        s_weights = [np.zeros(i.shape) for i in self.weights]
        r_biases = [np.zeros(i.shape) for i in self.biases]
        r_weights = [np.zeros(i.shape) for i in self.weights]
            
        change_weights = [np.zeros(i.shape) for i in self.weights]
        change_biases = [np.zeros(i.shape) for i in self.biases]
        #first and second moments for biases and weights
        for epoch in range(n_epochs):
            for i in range(n_batches):
                # Select the minibatch
                if i == n_batches - 1:
                    X_batch = x_scaled[i * M :, :]
                    y_batch = y_scaled[i * M :, :]
                else:
                    X_batch = minibatches[i][0]
                    y_batch = minibatches[i][1]

        # Feedforward and backpropagation
                self.feedforward(X_batch)
                nabla_b, nabla_w = self.backprop(X_batch, y_batch, lmb)
        # Update the moments for biases and weights
                s_biases = [rho1 * j + (1 - rho1) * i for j, i in zip(s_biases, nabla_b)]
                s_weights = [rho1 * j + (1 - rho1) * i for j, i in zip(s_weights, nabla_w)]

                r_biases = [rho2 * j + (1 - rho2) * (i ** 2) for j, i in zip(r_biases, nabla_b)]
                r_weights = [rho2 * j + (1 - rho2) * (i ** 2) for j, i in zip(r_weights, nabla_w)]

        # Bias correction
                s_biases_corrected = [j / (1 - rho1 ** (epoch + 1)) for j in s_biases]
                s_weights_corrected = [j / (1 - rho1 ** (epoch + 1)) for j in s_weights]
                r_biases_corrected = [j / (1 - rho2 ** (epoch + 1)) for j in r_biases]
                r_weights_corrected = [j / (1 - rho2 ** (epoch + 1)) for j in r_weights]

        # Compute the changes in parameters
                change_biases = [-(eta * j) / (np.sqrt(k) + delta) for j, k in zip(s_biases_corrected, r_biases_corrected)]
                change_weights = [-(eta * j) / (np.sqrt(k) + delta) for j, k in zip(s_weights_corrected, r_weights_corrected)]

        # Update parameters
                self.biases = [i + j for i, j in zip(self.biases, change_biases)]
                self.weights = [i + j for i, j in zip(self.weights, change_weights)]

    # Keep track of convergence condition
            if not self.classification:
                y_pred = self.feedforward(x_scaled)
                
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                    
                newscore = cost_function(y_pred)
                
            else: 
                y_prednew = np.where(self.feedforward(x_scaled) > threshold, 1, 0)
                #print(ypred == y_prednew)
                ypred=y_prednew
                newscore = self.accuracy(x_scaled, y)
            
            if not self.classification and abs(score-newscore)<=tol:
                score = newscore
                print(f"Convergence reached after {epoch + 1} epochs.")
                break;
          
            score = newscore 





#---------------- Breast cancer data -------------#
"""
For binary classification like the breast cancer data, use CostLogReg as
your cost function. For Multiclass classification, use CostCrossEntropy

This here is only a very tentative example of a grid search using Adam. I will
update the other methods and look into potential bugs with scaling/score calculation
at a later date.

For classification problems, the tolerance check is ignored. It often stops the training 
too early because of 1 single step that is too small.

ACCURACY NOW GIVES A TUPLE OF 6 ELEMENTS:
    
    0. Accuracy score;
    1. Rate of true positives;
    2. Rate of true negatives;
    3. Rate of false positives;
    4. Rate of false negatives.
    5. The full prediction;
"""



# Create heatmap
def plot_heatmap(accuracy, var1, var2, title, lablesize=15, fontsize=18, vmin=0, saveplot=False, test_lmb=False):
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy, annot=True, cmap="coolwarm", xticklabels=var2, yticklabels=var1, fmt='.3f', vmin=vmin)
    
    # Set original labels and title
    if title == "GD" or title == "Momentum-GD" or test_lmb==True:
        plt.xlabel(r" Learning Rate, $\eta$ []", fontsize = lablesize)
        plt.ylabel(r" L2-penalty, $\lambda$ []", fontsize = lablesize)
    else:
        plt.xlabel(r"Learning Rate, $\eta$ []", fontsize = lablesize)
        plt.ylabel(r"Number of minibatches, m []", fontsize = lablesize)
    
    plt.title(f"Heatmap of accuracy values for {title}\n ", fontsize=fontsize)
    plt.tight_layout()
    if saveplot == True:
        plt.savefig("Additional_Plots/" + title + "_heatmap.png")
    plt.show()
    
    # ------------------- Find index for  --------------
    max_val = np.max(accuracy)
    for i in range(len(accuracy[:,0])):
        for j in range(len(accuracy[0,:])):
            if accuracy[i,j] == max_val:
                i_max, j_max = i, j
    
    return i_max, j_max



def plot_confusion_matrix(y, y_pred, method_name, eta, m, lmd, act_func_name='deez nuts', num_nodes=-1, hidden_layers=1, logreg_plot=False, no_subtitle=False):
    # Confusion matrix data
    confusion = confusion_matrix(y, y_pred, normalize='true')
    
    fontsize = 18
    figsize = (6,6)
    lablesize = 15
      
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(confusion, annot=True, fmt='.3f', cmap='Blues', cbar=False, 
                xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'], ax=ax)
    
    # Main title
    plt.suptitle(f"Confusion Matrix - {method_name}", fontsize=fontsize)
    
    # Smaller subtitle
    if (logreg_plot == True) and (no_subtitle == False):
        ax.set_title(f"$\\eta$ = {eta}, $m$ = {m}, $\\lambda$ = {lmd}", fontsize=lablesize - 5 )
    if (logreg_plot == False) and (no_subtitle == False):
        ax.set_title(f"$\\eta$ = {eta}, $m$ = {m}, Activation Func. = {act_func_name} \n # nodes = {num_nodes}, # hidden layers = {hidden_layers}, $\\lambda$ = {lmd}", fontsize=lablesize - 5 )#, loc='center', pad=20)
    
    ax.set_xlabel("Predicted Label", fontsize=lablesize)
    ax.set_ylabel("True Label", fontsize=lablesize)
    
    # Display the plot
    plt.tight_layout()
    plt.savefig("Additional_Plots/" + method_name + "_confusion_matrix.png")
    plt.show()




if __name__ == "__main__":
    # ---------------- NOTES -------------------
    # 
    # figsize=(8,8)
    #
    # fontsize = fontsize
    # 
    # -------------------------------------------
    
    fontsize = 18
    figsize = (6,6)
    lablesize = 15
    
    
    
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=False)
    
    y = y.reshape(-1,1)
    
    """
    MLPGD = Network([30,100,1], LRELU, sigmoid, CostLogReg)
    MLPGD.reset_weights()
    MLPGD.set_classification()
    
    MLPMomGD = Network([30,100,1], LRELU, sigmoid, CostLogReg)
    MLPMomGD.reset_weights()
    MLPMomGD.set_classification()
    
    
    
    #Test to prove that lambda = 0 is the best
    eta_vals = np.logspace(-2, 0, 11)
    eta_vals = np.round(eta_vals, 2)
    lmbd_vals = np.logspace(-5-1, 0, 6+1)
    #add another value to lmbd_vals that is rewritten to zero
    lmbd_vals[0] = 0
    accuracy_listGD = np.zeros( (len(eta_vals), len(lmbd_vals)) )
    accuracy_listMomGD = np.zeros( (len(eta_vals), len(lmbd_vals)) )
    for i, eta in tqdm(enumerate(eta_vals)):
        print(f"eta number {i+1} of {len(eta_vals)}")
        for j, lmbd in enumerate(lmbd_vals):
            try:
                accuracyGD = MLPGD.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'GD', scale_bool = True, tol = 1e-17)
                accuracyMomGD = MLPMomGD.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0.9, method = 'GD', scale_bool = True, tol = 1e-17)
                accuracy_listGD[i][j] = accuracyGD[0]
                accuracy_listMomGD[i][j] = accuracyMomGD[0]
                MLPGD.reset_weights()
                MLPMomGD.reset_weights()
            except RuntimeWarning:
                MLPGD.reset_weights()
                MLPMomGD.reset_weights()
                continue;  
        
    
    
    iGD, jGD = plot_heatmap(accuracy_listGD.T, lmbd_vals, eta_vals, 'GD', saveplot=True, vmin=0.85)
    iMomGD, jMomGD = plot_heatmap(accuracy_listMomGD.T, lmbd_vals, eta_vals, 'Momentum-GD', saveplot=True, vmin = 0.85)
    
    
    #By inspecting these plots we determine the optimal lambda, which will be used in
    #RMSprop which we assume has the same optimal lambda.
    #Momentum GD tells us that a very small lambda is optimal.
    #Will choose 0 since it coincides with findings from other subtasks.
    
    #After testing RMSprop the actual optimal lambda is 0, as found in Mom GD plot
    
    
    
    hidden_nodes_vals = np.linspace(50, 150, 11)
    activation_funcs = [sigmoid, softmax, RELU, LRELU]
    act_funcs_labels = ['sigmoid', 'softmax', 'RELU', 'LRELU']
    accuracy_list_config_test = np.zeros( (len(hidden_nodes_vals), len(activation_funcs)) )
    for i, hidden in tqdm(enumerate(hidden_nodes_vals)):
        print(f"Number of nodes in hidden layer value {i+1} of {len(hidden_nodes_vals)}")
        for j, func in enumerate(activation_funcs):
            try:
                MLP_config_test = Network([30, int(hidden), 1], func, sigmoid, CostLogReg)
                MLP_config_test.reset_weights()
                MLP_config_test.set_classification()
                #picked some values that we knew were good from analysis that happens later in the code
                accuracy_config_test = MLP_config_test.fit(X, y, n_batches=32, n_epochs=100, eta=0.001, lmb=0, delta_mom=0, method='RMSprop', scale_bool = True, tol = 1e-17)
                accuracy_list_config_test[i][j] = accuracy_config_test[0]
            except RuntimeWarning:
                continue
    
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_list_config_test.T, annot=True, cmap="coolwarm", xticklabels=hidden_nodes_vals, yticklabels=act_funcs_labels, fmt='.3f')
    plt.xlabel(r" # of nodes in hidden layer []", fontsize = lablesize)
    plt.ylabel(r" Activation functions", fontsize = lablesize)
    plt.title(f"Heatmap of acc. values for RMSprop (network config.)\n ", fontsize=fontsize)
    #plt.tight_layout()
    plt.savefig("Additional_Plots/RMSprop_network_config_heatmap.png", pad_inches=0.5)
    plt.show()
    
    
    #Used m=32 and eta=0.001
    #According to this plot, we pick 100 nodes in hidden layer with RELU activation
    
    
    
    MLP_RMSprop1 = Network([30,100,1], RELU, sigmoid, CostLogReg)
    MLP_RMSprop1.reset_weights()
    MLP_RMSprop1.set_classification()
    
    MLP_AdaGrad1 = Network([30,100,1], RELU, sigmoid, CostLogReg)
    MLP_AdaGrad1.reset_weights()
    MLP_AdaGrad1.set_classification()
    
    MLP_ADAM1 = Network([30,100,1], RELU, sigmoid, CostLogReg)
    MLP_ADAM1.reset_weights()
    MLP_ADAM1.set_classification()
    
    MLP_RMSprop2 = Network([30,100,1], LRELU, sigmoid, CostLogReg)
    MLP_RMSprop2.reset_weights()
    MLP_RMSprop2.set_classification()
    
    MLP_AdaGrad2 = Network([30,100,1], LRELU, sigmoid, CostLogReg)
    MLP_AdaGrad2.reset_weights()
    MLP_AdaGrad2.set_classification()
    
    MLP_ADAM2 = Network([30,100,1], LRELU, sigmoid, CostLogReg)
    MLP_ADAM2.reset_weights()
    MLP_ADAM2.set_classification()
    
    m_list = np.linspace(10, 100, 10)
    #eta_vals = np.logspace(-3, -2, 11)
    eta_vals = np.logspace(-5, 0, 11)
    #eta_vals = np.logspace(eta_vals, 3)
    eta_vals = np.round(eta_vals, 5)
    
    accuracy_list_RMSprop1 = np.zeros( (len(eta_vals), len(m_list)) )
    accuracy_list_AdaGrad1 = np.zeros( (len(eta_vals), len(m_list)) )
    accuracy_list_ADAM1    = np.zeros( (len(eta_vals), len(m_list)) )
    accuracy_list_RMSprop2 = np.zeros( (len(eta_vals), len(m_list)) )
    accuracy_list_AdaGrad2 = np.zeros( (len(eta_vals), len(m_list)) )
    accuracy_list_ADAM2    = np.zeros( (len(eta_vals), len(m_list)) )
    
    for i, eta in tqdm(enumerate(eta_vals)):
        print(f"eta number {i+1} of {len(eta_vals)}")
        for j, m in enumerate(m_list):
            try:
                accuracy_RMSprop1 = MLP_RMSprop1.fit(X, y, n_batches = m, n_epochs = 100, eta = eta, lmb = 0, delta_mom = 0, method = 'RMSprop', scale_bool = True, tol = 1e-17)
                accuracy_AdaGrad1 = MLP_AdaGrad1.fit(X, y, n_batches = m, n_epochs = 100, eta = eta, lmb = 0, delta_mom = 0, method = 'Adagrad', scale_bool = True, tol = 1e-17)
                accuracy_ADAM1    = MLP_ADAM1.fit(X, y,    n_batches = m, n_epochs = 100, eta = eta, lmb = 0, delta_mom = 0, method = 'Adam',    scale_bool = True, tol = 1e-17)
                accuracy_list_RMSprop1[i][j] = accuracy_RMSprop1[0]
                accuracy_list_AdaGrad1[i][j] = accuracy_AdaGrad1[0]
                accuracy_list_ADAM1[i][j]    = accuracy_ADAM1[0]
                MLP_RMSprop1.reset_weights()
                MLP_AdaGrad1.reset_weights()
                MLP_ADAM1.reset_weights()
                
                accuracy_RMSprop2 = MLP_RMSprop2.fit(X, y, n_batches = m, n_epochs = 100, eta = eta, lmb = 0, delta_mom = 0, method = 'RMSprop', scale_bool = True, tol = 1e-17)
                accuracy_AdaGrad2 = MLP_AdaGrad2.fit(X, y, n_batches = m, n_epochs = 100, eta = eta, lmb = 0, delta_mom = 0, method = 'Adagrad', scale_bool = True, tol = 1e-17)
                accuracy_ADAM2    = MLP_ADAM2.fit(X, y,    n_batches = m, n_epochs = 100, eta = eta, lmb = 0, delta_mom = 0, method = 'Adam',    scale_bool = True, tol = 1e-17)
                accuracy_list_RMSprop2[i][j] = accuracy_RMSprop2[0]
                accuracy_list_AdaGrad2[i][j] = accuracy_AdaGrad2[0]
                accuracy_list_ADAM2[i][j]    = accuracy_ADAM2[0]
                MLP_RMSprop2.reset_weights()
                MLP_AdaGrad2.reset_weights()
                MLP_ADAM2.reset_weights()
                
            except RuntimeWarning:
                MLP_RMSprop1.reset_weights()
                MLP_AdaGrad1.reset_weights()
                MLP_ADAM1.reset_weights()
                
                MLP_RMSprop2.reset_weights()
                MLP_AdaGrad2.reset_weights()
                MLP_ADAM2.reset_weights()
                
                continue;  
    print('\n')
    
    iRMSprop1_max, jRMSprop1_max = plot_heatmap(accuracy_list_RMSprop1.T, m_list, eta_vals, title='RMSprop (ReLU)', vmin=0.95, saveplot=True)
    iAdaGrad1_max, jAdaGrad1_max = plot_heatmap(accuracy_list_AdaGrad1.T, m_list, eta_vals, title='AdaGrad (ReLU)', vmin=0.95, saveplot=True)
    iADAM1_max, jADAM1_max       = plot_heatmap(accuracy_list_ADAM1.T,    m_list, eta_vals, title='ADAM (ReLU)',    vmin=0.95, saveplot=True)
    
    iRMSprop2_max, jRMSprop2_max = plot_heatmap(accuracy_list_RMSprop2.T, m_list, eta_vals, title='RMSprop (LReLU)', vmin=0.95, saveplot=True)
    iAdaGrad2_max, jAdaGrad2_max = plot_heatmap(accuracy_list_AdaGrad2.T, m_list, eta_vals, title='AdaGrad (LReLU)', vmin=0.95, saveplot=True)
    iADAM2_max, jADAM2_max       = plot_heatmap(accuracy_list_ADAM2.T,    m_list, eta_vals, title='ADAM (LReLU)',    vmin=0.95, saveplot=True)
    
    #By eye: best values: m=30, eta=0.0032 (24.10.2024)
    #By eye: best values: m=80, eta=0.0032 (25.10.2024, after changes to stochastic part)
    #There are however MULTIPLE possible combinations with 100% accuracy
    """
    
    
    MLP = Network([30,100,1], RELU, sigmoid, CostLogReg)
    MLP.reset_weights()
    MLP.set_classification()
    
    final_accuracy = MLP.fit(X, y, n_batches=10, n_epochs=100, eta=0.03162, lmb=0, delta_mom=0, method = 'Adam', scale_bool = True, tol = 1e-17)
    
    #The true positive, false positive etc. values from .fit() are the number of cases divided by
    #total cases. Therefore TP+TN+FP+FN=1
    #In the confusion matrix TP+FP=1 and TN+FN=1
    print("When interpreting the confusion matrix: Benign (no cancer) = 1, Malignant (cancer) = 0")
    print("Upper right: True=0, Predicted=0.   Upper left: True=0, Predicted=1")
    print("Lower right: True=1, Predicted=0.   Lower left: True=1, Predicted=1")
    print('\n')
    
    print("Neural Network: ADAM, m=10, n_epochs=100, eta=0.03162, lambda=0")
    print(f"Accuracy: {final_accuracy[0]:.3f}, TP: {final_accuracy[1]:.3f}, TN: {final_accuracy[2]:.3f}, FP: {final_accuracy[3]:.3f}, FN: {final_accuracy[4]:.3f}")
    
    y_pred = final_accuracy[-1]
    print(f"log_loss = {log_loss(y, y_pred):.3f}")
    print('\n')
    plot_confusion_matrix(y, y_pred, 'ADAM', eta=0.03162, m=10, lmd=0, act_func_name='ReLU', num_nodes=100)
    
    
    
    from sklearn.neural_network import MLPClassifier
    
    y = y.flatten()
    clf = MLPClassifier(max_iter=10000)
    clf.fit(X, y)
    y_predSKNN = clf.predict(X)
    accuracySKNN = clf.score(X, y)
    
    print("Neural Network (Scikit learn), all default values")
    print(f"Accuracy: {accuracySKNN:.3f}")
    print(f"log_loss = {log_loss(y, y_predSKNN):.3f}")
    plot_confusion_matrix(y, y_predSKNN, 'SK-learn NN', eta=-1, m=-1, lmd=-1, no_subtitle='True')
    print('\n')