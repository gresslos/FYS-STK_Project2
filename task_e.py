import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score
from sklearn.model_selection import  train_test_split 
import seaborn as sns

from task_d_OP import plot_heatmap, Network, CostLogReg, sigmoid

"""
copied LogisticRegression from "Exercises Week 42: Logistic Regression and Optimization, reminders 
from week 38 and week 40", Hjorth-Jensen, Morten
"""


class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, t1=50, gamma=0.1, lam=0, 
                 num_iterations=1000, n_epochs=1000, m=10, 
                 overflow=True):
        self.learning_rate = learning_rate
        self.t1 = t1
        self.gamma = gamma
        self.lam = lam
        
        self.num_iterations = num_iterations
        self.n_epochs = n_epochs
        self.m = m
        
        #boolean to change which version of sigmoid to use in case of overflow
        self.overflow = overflow
        
        self.beta_logreg = None
    
    
    def sigmoid1(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid2(self, z):
        return np.exp(z) / ( 1 + np.exp(z) )
    
    
    def GDfit(self, X, y):
        n_data, num_features = X.shape
        self.beta_logreg = np.zeros(num_features)
        
        for _ in range(self.num_iterations):
            linear_model = X @ self.beta_logreg
            if self.overflow == False:
                y_predicted = self.sigmoid1(linear_model)
            else:
                y_predicted = self.sigmoid2(linear_model)
            
            # Gradient calculation
            gradient = (X.T @ (y_predicted - y))/n_data + 2*self.lam*self.beta_logreg
            
            # Update beta_logreg
            self.beta_logreg -= self.learning_rate*gradient
    
    
    #this is the one function that is actually made from scratch
    def SGDfit(self, X, y):
        
        n_data, num_features = X.shape
        self.beta_logreg = np.zeros(num_features)
        M = int(n_data/self.m)
        
        t0 = self.t1 * self.learning_rate
        
        #WITHOUT replacement
        data_size = X.shape[0]
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        
        shuffled_inputs = X[indices]
        shuffled_targets = y[indices]
        
        change = 0.0
        for i in range(self.n_epochs):
            for j in range(self.m):
                
                #WITH replacement
                #random_index = M*np.random.randint(self.m)
                
                Xi = shuffled_inputs[j*M:(j+1)*M, :]
                yi = shuffled_targets[j*M:(j+1)*M]
                
                linear_model = Xi @ self.beta_logreg
                if self.overflow == False:
                    y_predicted = self.sigmoid1(linear_model)
                else:
                    y_predicted = self.sigmoid2(linear_model)
                
                # Gradient calculation with L2 penalty
                gradient = (Xi.T @ (y_predicted - yi))/n_data + 2*self.lam*self.beta_logreg
                
                # Update beta_logreg
                eta = t0 / ( (i*self.m+j) + self.t1 )
                new_change = eta * gradient + self.gamma * change
                self.beta_logreg -= new_change
                change = new_change
    
    
    def predict(self, X):
        linear_model = X @ self.beta_logreg
        
        if self.overflow == False:
            y_predicted = self.sigmoid1(linear_model)
        else:
            y_predicted = self.sigmoid2(linear_model)
        
        return np.array([1 if i >= 0.5 else 0 for i in y_predicted])



def plot_confusion_matrix(y, y_pred, method_name, eta, m, lmd, t1=10, act_func_name='deez nuts', num_nodes=-1, hidden_layers=1, logreg_plot=False, no_subtitle=False):
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
        ax.set_title(f"$\\eta$ = {eta:.4f}, $m$ = {m}, $\\lambda$ = {lmd:.4f}, $t_1$ = {t1}", fontsize=lablesize - 5 )
    if (logreg_plot == False) and (no_subtitle == False):
        ax.set_title(f"$\\eta$ = {eta:.4f}, $m$ = {m}, $t_1$ = {t1}, Activation Func. = {act_func_name} \n # nodes = {num_nodes}, # hidden layers = {hidden_layers}, $\\lambda$ = {lmd:.4f}", fontsize=lablesize - 5 )#, loc='center', pad=20)
    
    ax.set_xlabel("Predicted Label", fontsize=lablesize)
    ax.set_ylabel("True Label", fontsize=lablesize)
    
    # Display the plot
    plt.tight_layout()
    plt.savefig("Additional_Plots/" + method_name + "_confusion_matrix.png")
    plt.show()




if __name__ == "__main__":
    
    print("When interpreting the confusion matrix: Benign (no cancer) = 1, Malignant (cancer) = 0")
    print("Upper right: True=0, Predicted=0.   Upper left: True=0, Predicted=1")
    print("Lower right: True=1, Predicted=0.   Lower left: True=1, Predicted=1")
    print('\n')
    
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
    
    # Sample data
    X, y = load_breast_cancer(return_X_y=True)
    
    eta_vals = np.logspace(-6, -2, 11)
    eta_vals = np.round(eta_vals, 6)
    lmbd_vals = np.logspace(-6, 1, 11)
    lmbd_vals = np.round(lmbd_vals, 8)
    accuracy_listGD = np.zeros( (len(eta_vals), len(lmbd_vals)) )
    accuracy_listSGD = np.zeros( (len(eta_vals), len(lmbd_vals)) )
    for i, eta in enumerate(eta_vals):
        for j, lmb in enumerate(lmbd_vals):
            try:
                modelGD = LogisticRegression(learning_rate=eta, t1=10, gamma=0, lam=lmb,
                                           num_iterations=1000, n_epochs=100, 
                                           m=10, overflow=True
                                           )
                modelSGD = LogisticRegression(learning_rate=eta, t1=10, gamma=0, lam=lmb,
                                           num_iterations=1000, n_epochs=100, 
                                           m=10, overflow=True
                                           )
                modelGD.GDfit(X, y)
                modelSGD.SGDfit(X, y)
                predictionsGD = modelGD.predict(X)
                predictionsSGD = modelSGD.predict(X)
                accuracy_listGD[i][j] = accuracy_score(y, predictionsGD)
                accuracy_listSGD[i][j] = accuracy_score(y, predictionsSGD)
            except RuntimeWarning:
                continue;  
    
    iGD_max, jGD_max = plot_heatmap(accuracy_listGD.T, lmbd_vals, eta_vals, title='GD (LogReg)', vmin=0.8, saveplot=True, test_lmb=True)
    iSGD_max, jSGD_max = plot_heatmap(accuracy_listSGD.T, lmbd_vals, eta_vals, title='Stochastic GD (LogReg)', vmin=0.8, saveplot=True, test_lmb=True)


    
    final_model = LogisticRegression(learning_rate=0.003981, t1=10, gamma=0, lam=1.99526231,
                                     num_iterations=1000, n_epochs=100, 
                                     m=10, overflow=True
                                     )
    
    final_model.SGDfit(X, y)
    y_predLG = final_model.predict(X)
    print("Logistic Regression: eta=0.003981, lambda=1.99526231, t1=10, n_epochs=100, m=10")
    print(f"Test set accuracy with Logistic Regression: = {accuracy_score(y, y_predLG):.3f}")
    print(f"log_loss = {log_loss(y, y_predLG):.3f}")
    y = y.reshape(-1, 1)
    y_predLG = y_predLG.reshape(-1, 1)
    plot_confusion_matrix(y, y_predLG, 'SGD (LogReg)', eta=0.003981, m=10, lmd=1.99526231, logreg_plot=True)
    print('\n')



    #Compare to sklearn logistic regression
    import matplotlib.pyplot as plt
    from sklearn.model_selection import  train_test_split 
    from sklearn.linear_model import LogisticRegression
    
    import warnings
    warnings.filterwarnings('ignore')
    
    # Load the data
    cancer = load_breast_cancer()

    # Logistic Regression
    logreg = LogisticRegression(solver='lbfgs') #L2 penalty is default in LogisticRegression
    logreg.fit(X, y)
    y_predSK = logreg.predict(X)
    
    #from sklearn.preprocessing import LabelEncoder
    print("Scikit-learn: No parameters to tune")
    print("Test set accuracy with Logistic Regression (Scikit-Learn) = {:.3f}".format(logreg.score(X, y)))
    print(f"log_loss = {log_loss(y, y_predSK):.3f}")
    plot_confusion_matrix(y, y_predSK, 'SK-learn LogReg', eta=-1, m=-1, lmd=-1, no_subtitle=True)
    print('\n')



    #Compare to neural network without hidden layers to show that it is
    #equivalent to logistic regression
    
    #Network needs an activation function for the hidden layer, even when there 
    #are no hidden layers.
    #Made a trivial activation function because I don't want any funny business.
    #A nice gut check for the fact that there is no hidden layer is that the
    #result doesn't depend on the activation function given for the hidden layer.
    def linear(X):
        return X
    
    MLP = Network(sizes=[30,1], hiddenact=linear, outputact=sigmoid, costfunc=CostLogReg)
    MLP.reset_weights()
    MLP.set_classification()
    
    eta_vals = np.logspace(-2, 2, 11)
    eta_vals = np.round(eta_vals, 6)
    lmbd_vals = np.logspace(-7, 0, 11)
    lmbd_vals = np.round(lmbd_vals, 6)
    accuracy_listNN = np.zeros( (len(eta_vals), len(lmbd_vals)) )
    for i, eta in enumerate(eta_vals):
        for j, lmb in enumerate(lmbd_vals):
            try:
                accuracyNN = MLP.fit(X, y, n_batches = 10, n_epochs = 100, 
                                     eta = eta, t1=10, lmb = lmb, 
                                     delta_mom = 0, method = 'SGD', scale_bool = True, 
                                     tol = 1e-17)
                accuracy_listNN[i][j] = accuracyNN[0]
                MLP.reset_weights()
            except RuntimeWarning:
                MLP.reset_weights()
                continue;
    
    iNN_max, jNN_max = plot_heatmap(accuracy_listNN.T, lmbd_vals, eta_vals, title='Stochastic GD (NN-LR)', vmin=0.8, saveplot=True, test_lmb=True)
    
    
    accuracyNN = MLP.fit(X, y, n_batches = 10, n_epochs = 100, eta = 15.848932, t1=10, lmb = 0.000316, delta_mom = 0, method = 'SGD', scale_bool = True, tol = 1e-17)
    y_predNN = accuracyNN[-1]
    print("Neural Network without hidden layers: eta=15.848932, lambda = 0.000316, t1=10, n_epochs=100, m=10")
    print(f"Test set accuracy for Neural Network without hidden layers = {accuracy_score(y, y_predNN):.3f}")
    print(f"log_loss = {log_loss(y, y_predNN):.3f}")
    plot_confusion_matrix(y, y_predNN, 'SGD (NN-LogReg)', eta=15.848932, m=10, lmd=0.000316, hidden_layers=0, act_func_name='N/A', num_nodes='N/A')