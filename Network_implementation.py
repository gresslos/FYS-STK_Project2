# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:41:00 2024

@author: User
"""

#---------------------imports-------------------------------#
import autograd.numpy as np
from autograd import grad, elementwise_grad


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import sklearn.datasets



from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers, initializers           #This allows using whichever regularizer we want (l1,l2,l1_l2) and initializer (Randn, He, Xavier)




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
"""
The behaviour of this class will depend on the type of problem it is being used for,
which is established by the "classification" attribute - inside the network, this is
handled automatically

REGRESSION - StandardScaler for both inputs and targets, reverse said scaling for
outputs

BINARY - MinMaxScaler for inputs, no scaling for targets (already assumed to be given
as an array of 0 and 1). Predictions not scaled, their conversion to an array of 0 and
1 is handled within the network

MULTICLASS - MinMaxScaler for inputs, one hot encoding for targets. Predictions can also
be one hot encoded with the "rescale" function
"""
class Scaler:
    def __init__(self, classification):
        """
        Initializes a scaler for both the inputs, X, and the targets, y,
        that depends on the type of problem.
        
        classification = False -> regression
        
        classification = "Binary" -> binary classification
        
        classification = "Multiclass" -> multiclass classification
        """
        self.classification = classification
        if not self.classification:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
        else:
            self.scaler_X = MinMaxScaler()
            self.scaler_y = OneHotEncoder(sparse_output = False)
        

    def scaletrain(self, X, y):
        """
        Scales training data, and transforms the scalers. For binary problems,
        ignores the One_hot_encoder set in __init___
        """
        
        X_scaled = self.scaler_X.fit_transform(X)
        
        if not self.classification:    
            y_scaled = self.scaler_y.fit_transform(y)
            return X_scaled, y_scaled
        
        elif self.classification=="Binary":
            y_scaled = y
            return X_scaled, y_scaled
        
        elif self.classification == "Multiclass":
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
            return X_scaled, y_scaled
        
    def scaletest(self, X, y):
        """
        Scales testing data, without transforming the scalers
        """
        X_scaled = self.scaler_X.transform(X)
        
        if not self.classification:    
            y_scaled = self.scaler_y.transform(y)
            return X_scaled, y_scaled
        
        elif self.classification=="Binary":
            y_scaled = y
            return X_scaled, y_scaled
        
        elif self.classification == "Multiclass":
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1))
            return X_scaled, y_scaled
    
    def rescale(self, y_pred):
        """
        Rescale predictions (y_pred) to the original scale. Does nothing in binary
        problems.
        """
        if self.classification == "Binary":
            pass;
        else:
            return self.scaler_y.inverse_transform(y_pred)
            


#Class for neural networks

class Network(object):

    def __init__(self, sizes, hiddenact, outputact, costfunc, initialization, seed=2024): #taken from Michael Nielsen
        """
        sizes: a list with the number of neurons per each hidden layer
        
        hiddenact: sets the activation function for the hidden layers
        
        outputact: sets the activation function for the output layer
        
        costfunc: sets the cost function to be minimized
        
        initialization: sets the initialization method for the weights and biases
        
        RANDN - both weights and biases initialized with an N(0,1) distribution
        
        HE - weights initialized with a N(0, sqrt(2/n_prev_layer_neurons) ) at each layer, 
        biases initialized to 0 
        
        XAVIER- weights initialized with a N(0, sqrt( 6/(n_prev_layer_neurons + n_current_layer_neurons) ) ) at each layer, 
        biases initialized to 0 
        
        seed: default to 2024, for reproducibility
        """
        
        self.seed = seed #seed for reproducibility
        
        
        self.hiddenact = hiddenact
        self.outputact = outputact
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.initialization = initialization
        
        np.random.seed(self.seed)
        if initialization == "RANDN":
            self.biases = [np.random.randn(1, y)  for y in sizes[1:]] 
            self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
            
        elif initialization == "HE":
            self.biases = [np.zeros((1, y))  for y in sizes[1:]] 
            self.weights = [np.random.randn(x, y)*np.sqrt(2/x) for x, y in zip(sizes[:-1], sizes[1:])]
            
        elif initialization == "XAVIER":
            self.biases = [np.zeros((1, y))  for y in sizes[1:]] 
            self.weights = [np.random.uniform(-np.sqrt(6/(x+y)), np.sqrt(6/(x+y)), size = (x, y)) for x, y in zip(sizes[:-1], sizes[1:])]
        
        # ------ LECTURE NOTES FOR WEEK 42 ------ #
        self.a_matrices = list() # list to store all the activations, layer by layer
        self.z_matrices = list() # list to store all the z vectors, layer by layer
        self.costfunc = costfunc
        self.classification = None
        

        
        
    def feedforward(self, X): #taken from Michael Nielsen, restructured for batching using week 42's exercises
        """
        RETURNS: A (n_inputs, n_output_neurons) sized matrix. Represents the outputs
        of the neurons in the final layer for each input contained in the batch X
            
        Also updates self.a_matrices and self.z_matrices: these are lists where each element
        corresponds to one of the layers and is a (n_inputs, n_layer_neurons) shaped numpy array.
        These arrays store the neuron activations, z, and outputs, a, obtained for each input
        contained in the batch X
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
        """
        RETURNS: a  list of tuples "(nabla_b, nabla_w)" representing the
        gradients for the cost function C. Each element of the list corresponds
        to gradients in one of the layers in the network
        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #derivatives of activation functions (TAKEN FROM LECTURE NOTES)
        out_derivative = derivate(self.outputact)
        hidden_derivative = derivate(self.hiddenact)
        
        #Delta and gradients for output layer
        
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
    def fit(self, X, y, n_batches, n_epochs, eta, lmb, delta_mom, method, scale_bool = True, tol = 1e-8, threshold = 0.5, early_stopping = True, SGD_bool = True):
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
         
     
     if method == "GD":
         scores = self.GD(X, y, eta, lmb, delta_mom, n_epochs, tol, threshold, scale_bool, early_stopping)
     
     elif method == "SGD":
         scores = self.SGD(X, y, n_epochs, n_batches, eta, lmb, delta_mom, tol, threshold, scale_bool, early_stopping)
         
     elif method == "Adagrad":
         scores = self.Adagrad(X, y, n_epochs, eta, lmb, n_batches, delta_mom, tol, threshold, SGD_bool, scale_bool, early_stopping)
         
     elif method == "RMSprop":
         scores = self.RMSprop(X, y, n_epochs, eta, lmb, n_batches, tol, threshold, scale_bool, early_stopping)
         
     elif method == "Adam":
         scores = self.Adam(X, y, n_epochs, eta, lmb, n_batches, tol, threshold, scale_bool, early_stopping)
         
     cost_function_train = self.costfunc(y)    
     # Performs a prediction with the updated parameters
     #and returns the value of the cost function
     
     """
     For regression: 
     Returns the final MSE and R2 scores, and also a list of all MSE over epochs, 
     that we can pull from depending on what we want to plot
     
     For multiclass classifications:
     Returns the final score, the predictions and a list of all scores over epochs 
     
     For binary classifications:
     Returns the final score, the rate of TP, TN, FP and FN, the predictions (to construct the confusion matrix) 
     and the list of all scores over epochs
     """
     
     if not self.classification:
         y_pred=self.feedforward(X_scaled)
         
         if scale_bool:
             y_pred = scaler.rescale(y_pred)
             
         score = cost_function_train(y_pred)
         score2 = r2_score(y_pred, y)
         
         return (score, score2, scores) 
     
     elif self.classification == "Multiclass":
         score, predictions = self.accuracy(X_scaled, y_scaled)[0]
         return (score, predictions, scores)
     
     elif self.classification == "Binary": 
         score0 = self.accuracy(X_scaled, y_scaled)[0]
         score1 = self.accuracy(X_scaled, y_scaled)[1]
         score2 = self.accuracy(X_scaled, y_scaled)[2]
         score3 = self.accuracy(X_scaled, y_scaled)[3]
         score4 = self.accuracy(X_scaled, y_scaled)[4]
         predictions = self.accuracy(X_scaled, y_scaled)[5]

         return (score0, score1, score2, score3, score4, predictions, scores)

    #-------------- LECTURE NOTES FROM WEEK 42-------------#
    
    def reset_weights(self):
        """
        Description:
        ------------
            Resets/Reinitializes the weights in order to train the network for a new problem.
            
            For better reproducibility, it is recommended that this method be called even
            if it is the first time in the program that the network is being trained.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.initialization == "RANDN":
            self.biases = [np.random.randn(1, y)  for y in self.sizes[1:]] 
            self.weights = [np.random.randn(x, y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            
        elif self.initialization == "HE":
            self.biases = [np.zeros((1, y))  for y in self.sizes[1:]] 
            self.weights = [np.random.randn(x,y)*np.sqrt(2/x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            
        elif self.initialization == "XAVIER":
            self.biases = [np.zeros((1, y))  for y in self.sizes[1:]] 
            self.weights = [np.random.uniform(-np.sqrt(6/(x+y)), np.sqrt(6/(x+y)), size = (x, y)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def split_mini_batches (self, nbatches, X, y):
        
        """
        create a list with n minibatches that the data is split into 
        Data is shuffled before making the batches
        
        Used to be with replacement - made without replacement with the help of GPT
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
    
    def predict(self, X_train, y_train, X_test, y_test, scale_bool = True, threshold=0.5):
        """
        Performs prediction after training of the network has been finished.
             
        Useful for running on unseen test data.

        Returns the MSE, R2 and predictions for regression problems.
        
        Returns the same as self.fit for classification problems.
        """
        
        if scale_bool:
            X_scaled, y_scaled, scaler = self.scaletraining(X_train, y_train)
            
            X_testscaled, y_testscaled = scaler.scaletest(X_test, y_test)
        else:
            X_testscaled, y_testscaled = X_test, y_test 
            

        if not self.classification:
            predict = self.feedforward(X_testscaled)
            
            if scale_bool:
                predict = scaler.rescale(predict)
                
            costfunc = self.costfunc(y_test)
            score = costfunc(predict)
            score2 = r2_score(predict, y_test)
            return (score, score2, predict)
        else:
            accuracies = self.accuracy(X_testscaled, y_testscaled)
            
            return accuracies
    # ------------ From lecture notes ---------- #
    def accuracy(self, X, y, threshold = 0.5):
        """
        Determine the accuracy of the results obtained for binary and multiclass
        classification problems.
        
        Threshold is only applicable for binary problems, defines which values
        of an output in the range [0,1] are set to 0 or 1
        """
        y_pred = self.feedforward(X)
        
        
        if self.classification == "Multiclass":
            predictions = np.argmax(y_pred, axis=1)  # For multi-class classification
            return [np.mean(predictions == np.argmax(y, axis=1)), predictions]
        
        if self.classification == "Binary":
            predictions = np.where(y_pred > threshold, 1, 0)
            
            #Testing for correlations (true/false positive/negatives)
            
            # Calculate correlations
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
            Decides if FFNN acts as a binary classifier (True) og regressor (False),
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
    
    """
    Implemented largely with help from ChatGPT - check conversation logs for more details
    
    All methods update the self.weights and self.biases lists. Calling one of them
    once (usually using self.fit) with the right architecture setup, should suffice for training the network.
    
    They also return a list of the scores at different epochs.
    
    
    All methods are initiated by specifying:
        the inputs X and targets y
        
        the gradient descent hyperparameters (learning rate, regularization and momentum if applicable)
        
        other hyperparameters such as number of iterations/epochs and number of batches
        
        the tolerance for the convergence condition (default setting at 1e-8) and the threshold for binary
        classification (default setting at 0.5)
        
        some values such as SGD_bool, scale_bool, early_stopping allow the user to select if they want
        stochasticity/scaling/the use of a convergence condition. These are all set to "True" by default.
        
        
    """
    def GD (self, X, y, eta, lmb, delta_mom, Niter, tol, threshold, scale_bool, early_stopping):

        # Establishing the cost function to evaluate the loss at each epoch
        cost_function = self.costfunc(y)   
        
        scores = list()
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
            score = self.accuracy(x_scaled, y_scaled)[0]
            
        scores.append(score)
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
                    newscore = self.accuracy(x_scaled, y_scaled)[0]
                    
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
                    newscore = self.accuracy(x_scaled, y_scaled)[0]
                    
                scores.append(newscore)    
                if not self.classification and abs(score-newscore)<=tol and early_stopping:
                    score = newscore
                    break;
                score = newscore     
        
        return scores
    
    def SGD(self, X, y, n_epochs, n_batches, eta0, lmb, delta_mom, tol, threshold, scale_bool, early_stopping):
        """
        Slightly altered relative to task a - the old implementation of time decay got stuck with an MSE
        of 9.95 for the 1D function, regardless of eta or lambda - probably
        because the learning rate decayed so fast that the different initial
        values ended up becoming irrelevant, or because the optimization became
        trapped in a local minimum.
        """
        # Establishing the cost function to evaluate the loss at each epoch
        cost_function = self.costfunc(y)
        
        scores = list()
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
            score = self.accuracy(x_scaled, y_scaled)[0]
            
        scores.append(score)
        
        n_epochs = int(n_epochs)
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches

        minibatches, x_scaled_shuffled, y_scaled_shuffled = self.split_mini_batches(n_batches, x_scaled, y_scaled)

        t0 = 1  # Arbitrary t0

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
                        
            if not self.classification:
                y_pred = self.feedforward(x_scaled)
                
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                    
                newscore = cost_function(y_pred)
                
            else: 
                newscore = self.accuracy(x_scaled, y_scaled)[0]
                
            scores.append(newscore)
            if not self.classification and abs(score-newscore)<=tol and early_stopping:
                score = newscore
                break;
            score = newscore
            
        return scores   

    def Adagrad(self, X, y, n_epochs, eta, lmb, n_batches, delta_mom, tol, threshold, SGD_bool, scale_bool, early_stopping):
        
        # Establishing the cost function to evaluate the loss at each epoch
        cost_function = self.costfunc(y)   
        
        scores = list()
        
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
            score = self.accuracy(x_scaled, y_scaled)[0]
        
        scores.append(score)
        # AdaGrad parameter to avoid possible division by zero
        delta = 1e-8

        # ----------------- SGD - parameters ---------------
        if SGD_bool:
            n_batches = int(n_batches)
            minibatches, x_scaled_shuffled, y_scaled_shuffled = self.split_mini_batches(n_batches, x_scaled, y_scaled)
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
            
            if not self.classification:
                y_pred = self.feedforward(x_scaled)
                
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                    
                newscore = cost_function(y_pred)
                
            else: 
                newscore = self.accuracy(x_scaled, y_scaled)[0]
                
            scores.append(newscore)    
            if not self.classification and abs(score-newscore)<=tol and early_stopping:
                score = newscore
                break;

        return scores          
    def RMSprop(self, X, y, n_epochs, eta, lmb, n_batches, tol, threshold, scale_bool, early_stopping):

        # Establishing the cost function to evaluate the loss at each epoch
        
        cost_function = self.costfunc(y)   
        scores = list() #list to save the scores over epochs
        
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
            score = self.accuracy(x_scaled, y_scaled)[0]
            
        scores.append(score)
        # Value for parameter rho
        rho = 0.99
        # RMSprop parameter to avoid possible division by zero
        delta = 1e-8

        # ----------------- SGD - parameters ---------------
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches
        minibatches, x_scaled_shuffled, y_scaled_shuffled = self.split_mini_batches(n_batches, x_scaled, y_scaled)
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
            
            if not self.classification:
                y_pred = self.feedforward(x_scaled)
                
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                    
                newscore = cost_function(y_pred)
                
            else: 
                newscore = self.accuracy(x_scaled, y_scaled)[0]
                
            scores.append(newscore)    
            if not self.classification and abs(score-newscore)<=tol and early_stopping:
                score = newscore
                break;
            score = newscore    
            
        return scores


    
    def Adam(self, X, y, n_epochs, eta, lmb, n_batches, tol, threshold, scale_bool, early_stopping):

        # Establishing the cost function to evaluate the loss at each epoch
        
        cost_function = self.costfunc(y)   
        
        scores = list() #list to save the scores over epochs
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
            score = self.accuracy(x_scaled, y_scaled)[0]
            
        scores.append(score)
            
        # Value for parameters rho1 and rho2, see https://arxiv.org/abs/1412.6980
        rho1 = 0.9
        rho2 = 0.999
        # AdaGrad parameter to avoid possible division by zero
        delta  = 1e-8   
        
        # ----------------- SGD - parameters ---------------
        n_batches = int(n_batches)
        M = X.shape[0] // n_batches
        minibatches, x_scaled_shuffled, y_scaled_shuffled = self.split_mini_batches(n_batches, x_scaled, y_scaled)
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
            if not self.classification:
                y_pred = self.feedforward(x_scaled)
                
                if scale_bool:
                    y_pred = scaler.rescale(y_pred)
                    
                newscore = cost_function(y_pred)
                
            else: 
                newscore = self.accuracy(x_scaled, y_scaled)[0]
                
            scores.append(newscore)
            if not self.classification and abs(score-newscore)<=tol and early_stopping:
                score = newscore
                break;
          
            score = newscore 
        return scores 


#------------------------------TENSORFLOW IMPLEMENTATION---------------------------------#
# Lecture notes from week 43, improved with GPT


def NN_model(inputsize, n_layers, n_neuron, optimizer, eta, lamda, activation, initializer):
    model = Sequential()
    if initializer == 'XAVIER':
        initializer = initializers.GlorotUniform(seed = 2024)
    elif initializer == 'RANDN':
        initializer = initializers.RandomNormal(mean=0.0, stddev=1, seed=2024)
    elif initializer == 'HE':
        initializer = initializers.HeNormal(seed = 2024)  
        
    # Add an explicit Input layer
    model.add(Input(shape=(inputsize,)))
    
    for i in range(n_layers):
        model.add(Dense(n_neuron, activation=activation, kernel_regularizer=regularizers.l2(lamda), kernel_initializer = initializer))
    
    model.add(Dense(1, kernel_regularizer=regularizers.l2(lamda), kernel_initializer = initializer))  # 1 output, no activation for linear regression

    # Choose the optimizer
    if optimizer == "ADAGRAD":
        optimizerr = optimizers.Adagrad(learning_rate=eta, initial_accumulator_value=0, epsilon=1e-08)
    elif optimizer == "ADAM":   
        optimizerr = optimizers.Adam(learning_rate=eta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == "RMSPROP":   
        optimizerr = optimizers.RMSprop(learning_rate=eta, rho=0.9, momentum=0.0, epsilon=1e-08)
    elif optimizer == "SGD":   
        optimizerr = optimizers.SGD(learning_rate=eta, momentum=0.0)

    # Compile the model with mean_squared_error and R² score
    model.compile(loss='mean_squared_error', optimizer=optimizerr, metrics=['mse', 'R2Score'])

    return model    