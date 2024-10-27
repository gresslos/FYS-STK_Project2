# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:31:48 2024

@author: User
"""
#---------------------imports-------------------------------#
import autograd.numpy as np
from autograd import grad, elementwise_grad

from imageio.v2 import imread

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("error")
#---------------- Cost, activation functions (Lecture notes) -------------#
def CostOLS(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


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

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

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
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def scaletrain(self, X, y):
        # Scale both features (X) and target (y)
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        return X_scaled, y_scaled
    
    def scaletest(self, X, y):
        #Scale testing data in the same way as training data
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)
        return X_scaled, y_scaled
    
    def rescale(self, y_pred):
        # Rescale predictions (y_pred) to the original scale
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
                    cost_func_derivative = elementwise_grad(self.costfunc(y))
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
    def fit(self, X, y, n_batches, n_epochs, eta, lmb, delta_mom, method, scale_bool, tol, X_val = None, y_val = None, SGD_bool = True):
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
         self.SGD(X, y, n_epochs, n_batches, tol, eta, lmb, delta_mom, scale_bool)
         
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
         score1 = mean_squared_error(y_pred, y)
         score2 = r2_score(y_pred, y)
         
         return (score1, score2)
     
     else: 
         score = self.accuracy(X_scaled, y)
         
         return score
     
    
    #-------------- LECTURE NOTES FROM WEEK 42 ------------------#
    def predict(self, X_train, y_train, X_test, y_test, scale_bool, threshold=0.5):
        """
         Description:
         ------------
             Performs prediction after training of the network has been finished.
             
             Useful for running on unseen test data.

         Parameters:
        ------------
             I   X (np.ndarray): The design matrix, with n rows of p features each

         Optional Parameters:
         ------------
             II  threshold (float) : sets minimal value for a prediction to be predicted as the positive class
                 in classification problems

         Returns:
         ------------
             I   z (np.ndarray): A prediction vector (row) for each row in our design matrix
                 This vector is thresholded if regression=False, meaning that classification results
                 in a vector of 1s and 0s, while regressions in an array of decimal numbers

        """
        
        if scale_bool:
            X_scaled, y_scaled, scaler = self.scaletraining(X_train, y_train)
            
            X_testscaled, y_testscaled = scaler.scaletest(X_test, y_test)
        else:
            
            X_testscaled = X_test 
            
        predict = self.feedforward(X_testscaled)
        if scale_bool:
            predict = scaler.rescale(predict)
    
        if self.classification:
            return np.where(predict > threshold, 1, 0)
        else:
            return predict
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
        if want_replacement:
            M = X.shape[0] // nbatches #size of minibatches
            batches=[]
        
            for i in range(nbatches):
                X_batch, y_batch = resample(X, y, replace= True, n_samples=M)
                batches.append((X_batch, y_batch))
            return batches, X, y
        else:
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
        # Scale values
        scaler = Scaler() # Create a scaler instance

        # Scale the input data and target
        x_scaled, y_scaled = scaler.scaletrain(x, y)
        return x_scaled, y_scaled, scaler #save the scaler to invert
    
    # ------------ From lecture notes ---------- #
    def set_classification(self):
        """
        Description:
        ------------
            Decides if FFNN acts as classifier (True) og regressor (False),
            sets self.classification during init()
        """
        self.classification = False
        if (self.costfunc.__name__ == "CostLogReg"
            or self.costfunc.__name__ == "CostCrossEntropy"):
            self.classification = True
    #------------------- GRADIENT DESCENT METHODS ---------------#
    
    # Implemented largely with help from chatGPT - check the conversation logs
    #for more detailed info
    
    def GD (self, X, y, eta, lmb, delta_mom, scale_bool, Niter, tol):

        cost_function = self.costfunc(y)   
        
        # Scaling the data if needed
        if scale_bool:
            x_scaled, y_scaled, scaler = self.scaletraining(X, y)
        else:
            x_scaled, y_scaled = X, y
        
        y_pred = self.feedforward(x_scaled)
        if scale_bool:
            y_pred = scaler.rescale(y_pred)
        
        if not self.classification:
            score = cost_function(y_pred)
             
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
                self.weights = [b + cb for b,cb in zip(self.biases, change_biases)]       
                
                # Check convergence after each iteration                           
                y_pred = self.feedforward(x_scaled)
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                
                newscore = cost_function(y_pred)
                
                if abs(score-newscore)<=tol:
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
                y_pred = self.feedforward(x_scaled)
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                
                newscore = cost_function(y_pred)
                
                if abs(score-newscore)<=tol:
                    score = newscore
                    print(f"Convergence reached after {iter} iterations.")
                    break;
                score = newscore     

    def SGD(self, x, y, n_epochs, n_batches, tol, eta0, lmb, delta_mom, scale_bool):
        
        # Establishing the cost function to evaluate the loss at each epoch
        
        cost_function = self.costfunc(y)   
        
        # Scaling the data if needed
        if scale_bool:
            x_scaled, y_scaled, scaler = self.scaletraining(X, y)
        else:
            x_scaled, y_scaled = X, y
        
        y_pred = self.feedforward(x_scaled)
        if scale_bool:
            y_pred = scaler.rescale(y_pred)
            
        score = cost_function(y_pred)

        n_epochs = int(n_epochs)
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches

        minibatches, x_scaled_shuffled, y_scaled_shuffled = self.split_mini_batches(n_batches, x_scaled, y_scaled) # saves the batches and the reshuffled X and y

        t0 = 1  # Arbitrary t0
        """
        The old implementation of time decay got stuck with an MSE
        of 9.95 for the 1D function, regardless of eta or lambda - probably
        because the learning rate decayed so fast that the different initial
        values ended up becoming irrelevant, or because the optimization became
        trapped in a lower minimum.
        """
        # Learning rate decay function (Inverse time decay)
        def time_decay_eta(eta0, t):
            return eta0 / (1 + t/t0)

        for epoch in range(n_epochs):
            #print(epoch, score)
            change_weights = [np.zeros(i.shape) for i in self.weights]
            change_biases = [np.zeros(i.shape) for i in self.biases]
            for i in range(n_batches):
                # Learning rate schedule based on time decay
                eta = time_decay_eta(eta0, epoch * n_batches + i)

                # Choose a minibatch and calculate gradients
                if i == n_batches - 1:
                    X_batch = x_scaled_shuffled[i * M:, :]
                    y_batch = y_scaled_shuffled[i * M:, :]
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
                        
            y_pred = self.feedforward(x_scaled)
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
            
            newscore = cost_function(y_pred)
            
            if abs(score-newscore)<=tol:
                score = newscore
                print(f"Convergence reached after {epoch + 1} epochs.")
                break;
            score = newscore
            

    def Adagrad(self, X, y, n_epochs, eta, lmb, n_batches, tol, delta_mom, SGD_bool, scale_bool):
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
        
        y_pred = self.feedforward(x_scaled)
        if scale_bool:
            y_pred = scaler.rescale(y_pred)
            
        score = cost_function(y_pred)
        
        # AdaGrad parameter to avoid possible division by zero
        delta = 1e-8

        # ----------------- SGD - parameters ---------------
        if SGD_bool:
            n_batches = int(n_batches)
            minibatches, x_scaled_shuffled, y_scaled_shuffled = self.split_mini_batches(n_batches, x_scaled, y_scaled) # saves the batches and the reshuffled X and y
        else:
            n_batches = 1
            x_scaled_shuffled, y_scaled_shuffled = x_scaled, y_scaled
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
                    X_batch = x_scaled_shuffled[i * M:, :]
                    y_batch = y_scaled_shuffled[i * M:, :]
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
            
            
            y_pred = self.feedforward(x_scaled)
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
            
            newscore = cost_function(y_pred)
            
            if abs(score-newscore)<=tol:
                score = newscore
                #print(f"Convergence reached after {epoch + 1} epochs.")
                break;
            score = newscore    

        #print("Training complete.")
                  
    def RMSprop(self, X, y, n_epochs, eta, lmb, n_batches, tol, scale_bool):
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
        y_pred = self.feedforward(x_scaled)
        if scale_bool:
            y_pred = scaler.rescale(y_pred)
          
        
        score = cost_function(y_pred)
        
        # Value for parameter rho
        rho = 0.99
        # RMSprop parameter to avoid possible division by zero
        delta = 1e-8

        # ----------------- SGD - parameters ---------------
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches
        minibatches, x_scaled_shuffled, y_scaled_shuffled = self.split_mini_batches(n_batches, x_scaled, y_scaled) # saves the batches and the reshuffled X and y
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
                    X_batch = x_scaled_shuffled[i * M:, :]
                    y_batch = y_scaled_shuffled[i * M:, :]
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
            
            
            y_pred = self.feedforward(x_scaled)
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
            
            newscore = cost_function(y_pred)
            
            if abs(score-newscore)<=tol:
                score = newscore
                print(f"Convergence reached after {epoch + 1} epochs.")
                break;
            score = newscore    
        



    def Adam(self, X, y, n_epochs, eta, lmb, n_batches, tol, scale_bool):

        # Establishing the cost function to evaluate the loss at each epoch
        
        cost_function = self.costfunc(y)   
        
        # Scaling the data if needed
        if scale_bool:
            x_scaled, y_scaled, scaler = self.scaletraining(X, y)
        else:
            x_scaled, y_scaled = X, y
        
        y_pred = self.feedforward(x_scaled)
        if scale_bool:
            y_pred = scaler.rescale(y_pred)
            
        score = cost_function(y_pred)
    
        # Value for parameters rho1 and rho2, see https://arxiv.org/abs/1412.6980
        rho1 = 0.9
        rho2 = 0.999
        # AdaGrad parameter to avoid possible division by zero
        delta  = 1e-8   
        
        # ----------------- SGD - parameters ---------------
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches
        minibatches, x_scaled_shuffled, y_scaled_shuffled = self.split_mini_batches(n_batches, x_scaled, y_scaled) # saves the batches and the reshuffled X and y
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
                    X_batch = x_scaled_shuffled[i * M :, :]
                    y_batch = y_scaled_shuffled[i * M :, :]
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
            y_pred = self.feedforward(x_scaled)
            if scale_bool:
                y_pred = scaler.rescale(y_pred)
            
            newscore = cost_function(y_pred)
            
            if abs(score-newscore)<=tol:
                score = newscore
                print(f"Convergence reached after {epoch + 1} epochs.")
                break;
            score = newscore 
            
#------------------------------------- RESULTS -----------------------------#
want_neurons = False

want_gridsearch = True

want_1D = True

want_replacement = False

want_franke = False

want_bootstrap = False   
#------------------------- Testing with simple 1D function --------------------#
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
if want_1D or want_neurons:
    
    want_franke = False
    
    np.random.seed(2024)
    
    n = 1000
    x = 2*np.random.rand(n,1)
    y = 2 + x + 2*x**2 + np.random.randn(n,1)
    # Creating Design Matrix
    X = np.c_[x, x**2] # No Intecept given Scaling-process

if want_franke:
    if want_bootstrap:
        with open("Bootstrap LRELU NOREP.txt","w") as file:
            file.write(" ") #Reset the file if the program is re-ran
    want_neurons = False
    want_gridsearch = False #Safeguards in case I forget to manually switch them off
    #-------------- Taken from the op_task_e code from last project - bootstrap analysis ------------#
    def bootstrap_comp(x, y, z, n_bootstraps=100, mindeg=1, maxdeg=15, interval=1):

        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        X = np.vstack((x_flat,y_flat)).T
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)
        # Note: X[:,0] = x_flat  X[:,1] = y_flat
        
        deg = np.arange(mindeg, maxdeg+1, interval) #degrees of polynomial
        
        
        error_test    = np.zeros(len(deg))
        bias_test     = np.zeros(len(deg))
        variance_test = np.zeros(len(deg))
        
        error_train = np.zeros(len(deg))
        bias_train = np.zeros(len(deg))
        variance_train = np.zeros(len(deg))
        
        for i in tqdm( range(len(deg)) ):
            z_train = z_train.reshape(-1,1)
            z_test = z_test.reshape(-1,1)
            
        
            #same number of points in the x and y direction
            #creates an array with n_inputs lines and n_bootstraps columns
            z_pred = np.empty((X_test.shape[0], n_bootstraps))
            z_tilde = np.empty((X_train.shape[0], n_bootstraps)) #pred is for test data, tilde for train data
            
            # Making Design Matrix Phi 
            Phi_train = Design_Matrix_2D(deg[i], X_train)
            Phi_test = Design_Matrix_2D(deg[i], X_test)
            
            # Initiating a network with the correct number of input neurons, 100
            # hidden neurons and 1 output
            num_feats = int((i+1 + 1) *(i+1 + 2) / 2 - 1)
            
            MLP = Network([num_feats,100,1], LRELU, identity, CostOLS) 
            
            for j in range(n_bootstraps):
            
                #make resampled design matrix, targets (scaling is handled by the network fit)
                
                X_, z_ = resample(X_train, z_train)
                Phi_train_ = Design_Matrix_2D(deg[i], X_)

                
                #for every bootstrap, the network must be reset
                MLP.reset_weights()
                MLP.fit(Phi_train_, z_, n_batches = 10, n_epochs = 100, eta = 0.001, lmb = 0.01, delta_mom = 0, method = 'Adagrad', scale_bool = True, tol = 1e-6)
                

                # Evaluate the new model on the same testing and training data each time.
                # The j-th column of z_pred corresponds to the j-th bootstrap
                z_pred[:, j]  = MLP.predict(Phi_train_, z_, Phi_test, z_test, scale_bool = True).flatten()
                z_tilde[:, j] = MLP.predict(Phi_train_, z_, Phi_train, z_train, scale_bool = True).flatten()
            
            
            error_test[i]    = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
            print(f"degree {i}, MSE test {error_test[i]}")
            bias_test[i]     = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance_test[i] = np.mean( np.var(z_pred, axis=1, keepdims=True)                )
            
            error_train[i]    = np.mean( np.mean((z_train - z_tilde)**2, axis=1, keepdims=True) )
            print(f"degree {i}, MSE train {error_train[i]}")
            bias_train[i]     = np.mean( (z_train - np.mean(z_tilde, axis=1, keepdims=True))**2 )
            variance_train[i] = np.mean( np.var(z_tilde, axis=1, keepdims=True)                 )
            
            with open("Bootstrap LRELU NOREP.txt","a") as file:
                file.write(f"Degree {i}: MSE {error_test[i]}, Bias {bias_test[i]}, Variance {variance_test[i]}\n")
        
        
        
        plt.figure(dpi=200)
        plt.plot(deg, error_test, label='Mean Squared Error', color='red')
        plt.plot(deg, bias_test, label='Bias', color='blue')
        plt.plot(deg, variance_test, label='Variance', color='lime')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Error')
        plt.xticks(deg)
        plt.title('Bias-Variance Trade Off (LRELU activation)', fontsize=10)
        plt.legend()
        plt.grid()
        plt.savefig("Bias variance LRELU NOREP.png")
        plt.show()
    
        
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)
    
    
    
    """
    Use optimal learning rate, lambda and method from the 1D test - justify that it would take too long to grid search
    with the terrain data, we decided to tune the params on
    
    RMS prop with 0.0001 eta and 0.01 lambda
    """
    if want_bootstrap:
        bootstrap_comp(x, y, z)
        
    else:
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        
        X = np.vstack((x_flat,y_flat)).T

        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.25)

            
        z_train = z_train.reshape(-1,1)
        z_test = z_test.reshape(-1,1)  #they are only reshaped for the purposes of train test split
        deg_max = 16
        errors = list()
        for deg in range(1,deg_max):
            Phi_train = Design_Matrix_2D(deg, X_train)
            Phi_test  = Design_Matrix_2D(deg, X_test) 
        
            num_feats = int((deg + 1) *(deg + 2) / 2 - 1)
        
            MLP = Network([num_feats,100,1], sigmoid, identity, CostOLS) 
            MLP.set_classification()
            MLP.reset_weights()
            MLP.fit(Phi_train, z_train, n_batches = 10, n_epochs = 100, eta = 0.001, lmb = 0.01, delta_mom = 0, method = 'Adagrad', scale_bool = True, tol = 1e-6)
    
            z_pred = MLP.predict(Phi_train, z_train, Phi_test, z_test, scale_bool = True)
            MSE = mean_squared_error(z_test,z_pred)
            errors.append(MSE)
            print(f"MSE for Franke's Function with degree {deg}: {MSE}")
            
        plt.figure()
        plt.yscale('log')
        plt.plot(range(1,deg_max), errors, label='Mean Squared Error', color='red') 
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Error')
        plt.xticks(range(1,deg_max))
        plt.title('MSE for Franke (sigmoid activation)', fontsize=10)
        plt.legend()
        plt.grid()
        plt.savefig("No bootstrap sigmoid NOREP.png")
        plt.show()
        
    
#--------------------- Testing different numbers of neurons with no regular., eta = 0.001 ----------------------#

if want_neurons:
    
    want_gridsearch = False #Safeguards in case I forget to manually switch them off
    eta = 0.01
    lam = 0.0001
    
    MSEsSGD, R2sSGD = list(), list() 
    MSEsAdagrad, R2sAdagrad = list(), list()
    MSEsRMSprop, R2sRMSprop = list(), list()
    MSEsADAM, R2sADAM = list(), list()
    for i in range(50,111):
        MLP = Network([2,i,1], sigmoid, identity, CostOLS) #MLP with 1 hidden layer with 25-100 neurons
        
        MLP.reset_weights()
        try:
            MSESGD, R2SGD = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lam, delta_mom = 0, method = 'SGD', scale_bool = True, tol = 1e-6)
            MSEsSGD.append(MSESGD)
            R2sSGD.append(R2SGD)
        except RuntimeWarning:
            pass;
        
        MLP.reset_weights()
        try:    
            MSEAdagrad, R2Adagrad = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lam, delta_mom = 0.5, method = 'Adagrad', scale_bool = True, tol = 1e-6)
            MSEsAdagrad.append(MSEAdagrad)
            R2sAdagrad.append(R2Adagrad)
        except RuntimeWarning:
            pass;   
     
        MLP.reset_weights()
        try:    
            MSERMSprop, R2RMSprop = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lam, delta_mom = 0, method = 'RMSprop', scale_bool = True, tol = 1e-6)
            MSEsRMSprop.append(MSERMSprop)
            R2sRMSprop.append(R2RMSprop)
        except RuntimeWarning:
            pass;   
        
        MLP.reset_weights()    
        try:    
            MSEADAM, R2ADAM = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lam, delta_mom = 0, method = 'Adam', scale_bool = True, tol = 1e-6)
            MSEsADAM.append(MSEADAM)
            R2sADAM.append(R2ADAM)
        except RuntimeWarning:
            pass;
            
    plt.figure(0)
    plt.title('MSE vs. number of neurons, 10 batches, 100 epochs')
    plt.xlabel('Number of hidden neurons')
    plt.ylabel('Mean squared error')
    plt.plot(range(50,111), MSEsSGD, color = 'orange', label = 'SGD')
    plt.plot(range(50,111), MSEsAdagrad, color = 'blue', label = 'Adagrad')
    plt.plot(range(50,111), MSEsRMSprop, color = 'purple', label = 'RMS prop')
    plt.plot(range(50,111), MSEsADAM, color='red', label='ADAM')
    plt.grid()
    plt.savefig("Neurons plot MSE NOREP.png")
    plt.legend() 
    
    plt.figure(1)
    plt.title('R2 vs. number of neurons, 10 batches, 100 epochs')
    plt.xlabel('Number of hidden neurons')
    plt.ylabel('R2 score')
    plt.plot(range(50,111), R2sSGD, color = 'orange', label = 'SGD')
    plt.plot(range(50,111), R2sAdagrad, color = 'blue', label = 'Adagrad')
    plt.plot(range(50,111), R2sRMSprop, color = 'purple', label = 'RMS prop')
    plt.plot(range(50,111), R2sADAM, color='red', label='ADAM')
    plt.grid()
    plt.savefig("Neurons plot R2.png")
    plt.legend()
#--------------------- GRID SEARCH FOR OPTIMAL PARAMETERS --------------#

#--------------------- Grid search from Morten, plotting with help from GPT ----------------#

if want_gridsearch:
    eta_vals = np.logspace(-4, 1, 6)
    lmbd_vals = np.logspace(-5, 1, 7)
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
    
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            print(f"Eta: {eta}, lambda: {lmbd}")
            
            MLP = Network([2,100,1], sigmoid, identity, CostOLS) #MLP with 1 hidden layer with 100 neurons
            MLP.set_classification()
            MLP.reset_weights()
            
            try:
                MSESGD, R2SGD = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'SGD', scale_bool = True, tol = 1e-6)
                MSEsSGD[i][j] = MSESGD
                R2sSGD[i][j] = R2SGD
            except RuntimeWarning:
                MSEsSGD[i][j]=float("inf")
                R2sSGD[i][j]=float("inf")
                pass;
            
            MLP.reset_weights()
            try:    
                MSEAdagrad, R2Adagrad = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'Adagrad', scale_bool = True, tol = 1e-6)
                MSEsAdagrad[i][j] = MSEAdagrad
                R2sAdagrad[i][j] = R2Adagrad
            except RuntimeWarning:
                MSEsAdagrad[i][j]=float("inf")
                R2sAdagrad[i][j]=float("inf")
                pass;   
         
            MLP.reset_weights()
            try:    
                MSERMSprop, R2RMSprop = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'RMSprop', scale_bool = True, tol = 1e-6)
                MSEsRMSprop[i][j] = MSERMSprop
                R2sRMSprop[i][j] = R2RMSprop
            except RuntimeWarning:
                MSEsRMSprop[i][j]=float("inf")
                R2sRMSprop[i][j]=float("inf")
                pass;   
            
            MLP.reset_weights()    
            try:    
                MSEADAM, R2ADAM = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = eta, lmb = lmbd, delta_mom = 0, method = 'Adam', scale_bool = True, tol = 1e-6)
                MSEsADAM[i][j] = MSEADAM
                R2sADAM[i][j] = R2ADAM
                print(MSEADAM)
            except RuntimeWarning:
                MSEsADAM[i][j]=float("inf")
                R2sADAM[i][j] = float("inf")
                pass;


    # Assuming eta_vals and lmbd_vals are logarithmic ranges like np.logspace(-6, 0, 7) or similar


    # Plot heatmaps with NaN values highlighted (mean squared error)
    fig, ax = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsSGD)
    sns.heatmap(np.log10(MSEsSGD), annot=True, ax=ax, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'MSE (log 10)'})
    ax.set_title("MSE for SGD, 10 batches, 100 epochs")
    ax.set_ylabel("Learning rate")
    ax.set_xlabel("Lambda")
    plt.savefig("MSE SGD own sigmoid log.png")
    plt.show()
    
    
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsAdagrad)
    sns.heatmap(np.log10(MSEsAdagrad), annot=True, ax=ax1, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'MSE (log 10)'})
    ax1.set_title("MSE for Adagrad, 10 batches, 100 epochs")
    ax1.set_ylabel("Learning rate")
    ax1.set_xlabel("Lambda")
    plt.savefig("MSE Adagrad own sigmoid log.png")
    plt.show()
    

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsRMSprop)
    sns.heatmap(np.log10(MSEsRMSprop), annot=True, ax=ax2, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'MSE (log 10)'})
    ax2.set_title("MSE for RMSprop, 10 batches, 100 epochs")
    ax2.set_ylabel("Learning rate")
    ax2.set_xlabel("Lambda")
    plt.savefig("MSE RMSprop own sigmoid log.png")
    plt.show()
    

    fig3, ax3 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(MSEsADAM)
    sns.heatmap(np.log10(MSEsADAM), annot=True, ax=ax3, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'MSE (log 10)'})
    ax3.set_title("MSE for ADAM, 10 batches, 100 epochs")
    ax3.set_ylabel("Learning rate")
    ax3.set_xlabel("Lambda")
    plt.savefig("MSE ADAM own sigmoid log.png")
    plt.show()
    
    
    # Plot heatmaps with NaN values highlighted (R2 score)
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sSGD)
    sns.heatmap(R2sSGD, annot=True, ax=ax4, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'R2'})
    ax4.set_title("R2 for SGD, 10 batches, 100 epochs")
    ax4.set_ylabel("Learning rate")
    ax4.set_xlabel("Lambda")
    plt.savefig("R2 SGD own sigmoid.png")
    plt.show()
    
    
    fig5, ax5 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sAdagrad)
    sns.heatmap(R2sAdagrad, annot=True, ax=ax5, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'R2'})
    ax5.set_title("R2 for Adagrad, 10 batches, 100 epochs")
    ax5.set_ylabel("Learning rate")
    ax5.set_xlabel("Lambda")
    plt.savefig("R2 Adagrad own sigmoid.png")
    plt.show()
    
    
    fig6, ax6 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sRMSprop)
    sns.heatmap(R2sRMSprop, annot=True, ax=ax6, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'R2'})
    ax6.set_title("R2 for RMSprop, 10 batches, 100 epochs")
    ax6.set_ylabel("Learning rate")
    ax6.set_xlabel("Lambda")
    plt.savefig("R2 RMSprop own sigmoid.png")
    plt.show()
    

    fig7, ax7 = plt.subplots(figsize=(10, 10))
    mask = np.isinf(R2sADAM)
    sns.heatmap(R2sADAM, annot=True, ax=ax7, cmap="crest", mask=mask, xticklabels=lmbd_vals, yticklabels=eta_vals, cbar_kws={'label': 'R2'})
    ax7.set_title("R2 for ADAM, 10 batches, 100 epochs")
    ax7.set_ylabel("Learning rate")
    ax7.set_xlabel("Lambda")
    plt.savefig("R2 ADAM own sigmoid.png")
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
    with open("MSE_R2_own_sigmoid.txt","w") as file:
        file.write(" ") #Reset the file if the program is re-ran
    
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
    
    with open("MSE_R2_own_sigmoid.txt","a") as file:
        file.write(f"SGD: Best MSE = {optimalMSESGD}; Best R2 = {optimalR2SGD}; Best eta = {optimaletaSGD}; Best lambda = {optimalmbdSGD}")
        file.write(f"Adagrad: Best MSE = {optimalMSEAdagrad}; Best R2 = {optimalR2Adagrad}; Best eta = {optimaletaAdagrad}; Best lambda = {optimalmbdAdagrad}")
        file.write(f"RMSprop: Best MSE = {optimalMSERMSprop}; Best R2 = {optimalR2RMSprop}; Best eta = {optimaletaRMSprop}; Best lambda = {optimalmbdRMSprop}")
        file.write(f"ADAM: Best MSE = {optimalMSEADAM}; Best R2 = {optimalR2ADAM}; Best eta = {optimaletaADAM}; Best lambda = {optimalmbdADAM}")