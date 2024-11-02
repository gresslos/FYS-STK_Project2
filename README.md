# FYS-STK_Project2

## Overview
- `task_a.py`                           :  OLS, GD-methods, and plots of results on data: **3. order polymial**
- `Network_implementation.py`           :  Contains both the latest version of our Neural Network class and also a function for constructing a TensorFlow equivalent
- `task b c.py`                         :  Plots for the use of neural networks for regression tasks. The kind of analysis performed by the program is decided by user input upon initiating it. 
- `task_b_c_model_comparison.py`   :  Fits to the Franke Function training dataset of our own network implementation, an analytical OLS regression to a polynomial of degree 7 and a TensorFlow network with the same architecture. Compares all 3 by determining MSE on an unseen test data set. 
- `task d.py`                           :  **Write here**
- `task_d_OP.py`                        :  **Write here**
- `task_e.py`                           :  **Write here**
- Additional_Plots/Linear_Regression    : Include plots from `task_a.py`, `task b c.py`
- Additional_Plots/Network_Regression   : Include plots from `task b c.py`
- Additional_Plots/Classification       : Include plots from `task_d_OP.py` and `task_e.py`

  

## Functions
### GD-Functions (found in `task_a.py`)
- **Input Variables**:
  - `X`, `y`, `beta0`: Design Matrix, data point, regression coefficent respectively
  - `Niter`, `n_epochs`, `m`, `eta / eta0`, `lmb`, `tol`: Hyperparameters
  - bool-values: Defined under `__main__ == "__main__"` used to decide if you want a given property (momentum, stochasticity, etc.) or not
    - **Note**: To switch between analytical cost function gradient and automatic differentiation change `want_Autograd` (line 532).
- **Structure**: Detailed structure provided in `Project2.tex`.

### Activations and cost functions (found in `Network_implementation.py` and `task_d_OP.py`)
  
  Contains all of the hidden layer activations (`sigmoid`, `RELU`, `LRELU`), output activations (`sigmoid`, `softmax`, `identity`) and cost functions (`CostOLS`, `CostLogReg`, `CostCrossEntropy`) used in this project

### derivate (found in `Network_implementation.py` and `task_d_OP.py`)
- **Inputs**
  - `func`: the function to be differentiated, Callable
  
  Used to handle automatic differentiation with the 'elementwise_grad' function from autograd. For `RELU` and `LRELU`, determines the analytical gradient, to avoid problems with the non-differentiable point at X = 0.

- **Returns**: The gradient of the input, `func`, which will be another function with the same input variables.

### Design_Matrix_2D (found in `Network_implementation.py`)
- **Inputs**
  - `deg`: The degree of the design matrix one wants to produce, int
  - `X`: An array containing the (x,y) datapoints , numpy array of shape (n_inputs, 2)

  Produces a design matrix of degree `deg` for a 2D dataset. The matrix is created without an intercept column.

- **Returns**: `Phi`: The design matrix, numpy array of shape (n_inputs, n_features), where the n_features is given by (`deg`+1)(`deg`+2)/2 - 1  


## Classes

### Scaler (found in `Network_implementation.py`)
- **Initial conditions**:
  - `classification`: Three possible values: False for regression problems, "Binary" for binary classification, "Multiclass" for multiclass classification. If False, creates a StandardScaler() for both inputs and targets. Otherwise, creates a MinMaxScaler() for inputs and a OneHotEncoder() for targets.

- **Methods**:
  - `scaletrain`:
    - **Inputs**
      - `X`: training input data, numpy array of shape (n_inputs, n_features)
      - `y`: training target variables, numpy array of shape (n_inputs, 1)
   Trains the scalers defined when the class was initiated on a set of training data, while simultaneously transforming said data. For Binary classification problems, does nothing to the targets.

   Returns: the scaled training inputs and targets

  - `scaletest`: 
    - **Inputs**
      - `X`: test input data, numpy array of shape (n_inputs, n_features)
      - `y`: test target variables, numpy array of shape (n_inputs, 1)

   Scales a set of testing data according to how the scaler was previously trained. Should only be called after `scaletrain` has been used on a set of training data

   Returns: the scaled testing inputs and targets

  -`rescale`: 
    - **Inputs**
      - `y_pred`: predictions made on scaled data, numpy array of shape (n_inputs, 1)
      
   Rescales a prediction according to the trained targets scaler. Should only be called after `scaletrain` has been used on a set of training data. For classification == "Binary", does nothing (the Network class handles scaling of these predictions internally)

   Returns: the rescaled predictions

### Network (found in `Network_implementation.py`)
- **Initial conditions**:
  - `Sizes`: List of int values, defines the number of nodes in each layer. The first element of the list is the input nodes (should be the number of features in the data), whereas the last is the number of output nodes. Any values in between correspond to hidden layers
  - `hiddenact`: The activation function for the hidden layers, Callable
  - `outputact`: The activation function for the output layer, Callable
  - `costfunc`: The cost function to be minimized, Callable
  - `initialization`: The parameter initialization scheme to be used. Can either be "RANDN" (weights and biases initiated with an N(0,1) distribution), "XAVIER" (initialization scheme introduced by Xavier Glorot) or "HE" (initialization scheme introduced by Kaiming He), string
  - `seed`: The RNG seed of the network, default setting is 2024 for reproducing the results in the report, int
  - `weights` and `biases`: Lists of numpy arrays containing the weight matrices and bias vectors for each layer. Automatically initiated depending on the `Sizes` and `initialization` settings.
  - `z_matrices` and `a_matrices`: Initiated as empty lists, are filled once a feedforward step is completed. Created to store the activations and outputs that are obtained at each layer after feeding a batch of inputs forward through the network.
  - `classification`: Defines whether the network is being used as a regressor, a binary classifier or a multiclass classifier. This is important because the type of task affects some of the functions in our implementation. Is initiated as "None", can be defined by running the "set_classification" method.

- **Methods**:
  - `feedforward`: 
    - **Inputs** 
      - `X`: input data, numpy array of shape (n_inputs, n_features)

    Feeds a batch of input datapoints, X, through the network by successive application of affine transformations and activation functions at each layer. Stores the activations, z, and outputs, a, eat each layer as elements of self.z_matrices and self.a_matrices

    Returns: The outputs of the network
  
  - `backprop`: 
    - **Inputs**
      - `X`: input data, inputs, numpy array of shape (n_inputs, n_features)
      - `y`: target variables, numpy array of shape (n_inputs, 1)
      - `lmb`: L2 regularization hyperparameter for the weights, float

    Performs a backpropagation of the error through the network, starting at the output layer. Relies on the values of self.z_matrices and self.a_matrices, and should thus only be called after an instance of feedforward.

    Returns: A list of tuples "(nabla_b, nabla_w)" representing the
        gradients for the cost function C. Each element of the list corresponds
        to gradients in one of the layers in the network

  - `fit`: 
    - **Inputs**
      - `X`: input data, numpy array of shape (n_inputs, n_features)
      - `y`: target variables, numpy array of shape (n_inputs, 1)
      -  `n_batches`: number of batches that the data is split into, int
      - `n_epochs`: number of training epochs, int
      - `eta`: initial learning rate for the gradient descent methods, float
      - `lmb`: L2 regularization parameter for the weights, float
      - `delta_mom`: momentum parameter (only applicable to GD, SGD and AdaGrad), float
      - `method`: defines which gradient descent method to use, possible values are "GD", "SGD", "Adagrad", "RMSprop" and "Adam". string
      - `scale_bool`: default set to True, decides if scaling is used or not, boolean
      - `tol`: default set to 1e-8, defines how much the score must change between epochs for the training to keep going, float
      - `threshold`: (only applicable to binary classification), default set to 0.5, sets the value that separates 0 and 1 predictions in binary classification problems, float
      - `early_stopping`: default set to True, if set to False the early stopping criterium provided by tol is ignored and the training always lasts for the full number of epochs, boolean
      - `SGD_bool`: (only applicable to AdaGrad), default set to True, decides if stochastic gradient descent is used, boolean

           
      Trains the parameters of the network on a set of training data, utilizing one of the available gradient descent methods (GD, SGD, AdaGrad, RMSprop, ADAM)

      Returns: 
        For regression tasks (classification = False): the final MSE and R2 scores; a list of all MSE scores over epochs

        For binary classification tasks (classification = "Binary"): the final accuracy score; the rate of true positives, true negatives, false positives and false negatives; the predictions; a list of all accuracy scores over epochs

        For multiclass classification tasks (classification = "Multiclass"): the final accuracy score; the predictions; a list of all accuracy scores over epochs 

  - `reset_weights`: Resets the RNG seed to self.seed. Then restarts the weights and biases in the network according to the chosen initialization scheme. For reproducibility of results, it is recommended that this method always be called before any other operations are performed on the network.

  - `split_mini_batches`: 
    - **Inputs**
      - `nbatches`: number of batches that the data is split into, int
      - `X`: input data, numpy array of shape (n_inputs, n_features)
      - `y`: target variables, numpy array of shape (n_inputs, 1)

      Shuffles the inputs and targets and splits them into minibatches for use in stochastic gradient descent methods. The divisibility between the number of inputs and the number of batches is not an issue: the gradient descent methods within the network handle this problem internally

        Returns - minibatches: a list of tuples containing the different batches;
                  shuffled_inputs, shuffled_targets: the shuffled X and y arrays
                  
  - `scaletraining`: 
    - **Inputs**
      - `x`: input data, numpy array of shape (n_inputs, n_features)
      - `y`: target variables, numpy array of shape (n_inputs, 1)      
                                  

      Creates an instance of the Scaler class with the same "classification" value as the network itself. Then trains said scaler on the inputs and targets, while simultaneously scaling them.

        Returns - x_scaled, y_scaled: the scaled inputs and targets;
                  scaler: the trained Scaler

  - `predict`:  
    - **Inputs**
      - `X_train`: inputs of the training data
      - `y_train`: target variables of the training data
      - `X_test`:  inputs of the testing data 
      - `y_test`:  target variables of the testing data 
      - `scale_bool`: default set to True, decides if scaling is used or not, boolean
      - `threshold`: (only applicable to binary classification), default set to 0.5, sets the value that separates 0 and 1 predictions in binary classification problems, float 

      Performs a prediction after training of the network has been finished. The training data is only necessary in this case for the purposes of scaling the testing data and the predictions. This method should only be called after the network has been trained using 'fit'.

        Returns:
        
         For regression tasks (classification = False): the final MSE and R2 scores; the predictions

         For binary classification tasks (classification = "Binary"): the final accuracy score; the rate of true positives, true negatives, false positives and false negatives; the predictions

         For multiclass classification tasks (classification = "Multiclass"): the final accuracy score; the predictions

  - `accuracy`:  
    - **Inputs**
      - `X`: input data, numpy array of shape (n_inputs, n_features), 
      -  `y`: target variables, numpy array of shape (n_inputs, 1), 
      - `threshold`: (only applicable to binary classification), default set to 0.5, sets the value that separates 0 and 1 predictions in binary classification problems, float
                        
      Only usable for classification problems. Feeds the inputs forward through the network and then transforms the outputs into either a string of 0 and 1 values (for classification == "Binary") or  one-hot vectors (for classification == "Multiclass"), before comparing them with the targets. 
      
      Usually shouldn't be called on its own, but rather when utilizing other methods such as 'fit' or 'predict'.

        Returns:

         For binary classification tasks (classification = "Binary"): the accuracy score; the rate of true positives, true negatives, false positives and false negatives; the predictions

         For multiclass classification tasks (classification = "Multiclass"): the accuracy score; the predictions

  - `set_classification`:
                        
      Defines what kind of problem the network is trying to solve by checking its "costfunc" and "outputact" attributes. Should always be called before any other operations are performed on the network.
 
  - `GD`, `SGD`, `Adagrad`, `RMSprop`, `Adam`:

      The different gradient descent methods that can be used to train the network. They should not be called on their own, but rather by using the 'fit' method and selecting the desired 'method' argument. Their input variables were already explained in 'fit'.

                        
                      

Do similar as above OLP and JH for certain functions



## How to Run the Code
```bash
$ python3 <filename>