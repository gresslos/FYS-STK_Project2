import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

#import jax.numpy as jnp
#from jax import grad
import autograd.numpy as anp
from autograd import grad
    


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

np.random.seed(0)  # For reproducibility

def R2(y_data,y_model):
    return 1 - anp.sum((y_data - y_model) ** 2) / anp.sum((y_data - anp.mean(y_data)) ** 2)


# Defining our Cost-Function 
def CostOLS(y_data,X,beta):
    #n = np.size(y_data)
    n = len(y_data)
    return anp.sum((y_data-X @ beta)**2) / n

# Defining Analytical Gradient of Cost-Function
# See derivations Week 39 Lecture notes
def grad_anl(y_data, X, beta):
    n = np.size(y_data)
    return (2.0 / n)  *X.T @ (X @ beta - y_data)

# MSE funciton for easy comparisong data vs. model
def MSE(y_data, y_model):
    n = np.size(y_data)
    return np.sum((y_data - y_model)**2) / n



# Class for scaling and rescaling using StandardScaler
class Scaler:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def scale(self, X, y):
        # Scale both features (X) and target (y)
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        return X_scaled, y_scaled

    def rescale(self, y_pred):
        # Rescale predictions (y_pred) to the original scale
        return self.scaler_y.inverse_transform(y_pred)
    



def OLS(X, y):
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)

   # Make Prediction
    y_pred = X @ beta

    return y_pred




def GD(X,y, beta0, Niter, tol, eta, lmb=0):
    beta = np.copy(beta0) # Copy to not change beta0 for nest Regression model

    MSE_list = np.zeros(Niter+1)
    MSE_list[0] = MSE(y, X @ beta)


    diff = tol + 1
    iter = 0

    while iter < Niter and tol < diff:
        #new_change = eta * (grad_anl(y, X, beta) + 2 * lmb * beta)
        new_change = eta * (gradient(y, X, beta) + 2 * lmb * beta)
        beta -= new_change
        # Will be plain OLS if lmb = 0

        # Calculate MSE-value----------
        y_pred_i = X @ beta
        MSE_list[iter+1] = MSE(y,y_pred_i)
        # ----------------------------

        iter += 1
        diff = np.linalg.norm(new_change)

    if iter == Niter:
        iter = -1 #f"Did not convergence"

    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter, MSE_list



def momGD(X,y, beta0, Niter, tol, eta, delta_mom, lmb=0):
    beta = np.copy(beta0)

    MSE_list = np.zeros(Niter+1)
    MSE_list[0] = MSE(y, X @ beta)

    change = 0.0
    
    diff = tol + 1
    iter = 0
    while iter < Niter and tol < diff:
        new_change = eta * gradient(y,X,beta) + delta_mom * change # add momentum
        beta -= new_change                                         # make change
        change = new_change                                        # save change

        # Calculate MSE-value----------
        y_pred_i = X @ beta
        MSE_list[iter+1] = MSE(y,y_pred_i)
        # ----------------------------

        iter += 1
        diff = np.linalg.norm(new_change)

    if iter == Niter:
        iter = -1 #f"Did not convergence"

    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter, MSE_list


def SGD(X, y, beta, n_epochs, m,  tol, eta0, delta_mom, momentum = True):
    # - Use the best learning rate (eta) from GD / momGD - analyze 
    # - as tunable_eta(0)

    n_epochs = int(n_epochs)
    m = int(m)
    beta = np.copy(beta0)

    MSE_list = np.zeros(Niter+1)
    MSE_list[0] = MSE(y, X @ beta)

    
    # t0 / t1 = eta_best from GD
    # PS: chose t1 = 10 -> will give t0
    # Impact scaling: t compared to t1
    t1 = 50
    t0 =  eta0 * t1
    def time_decay_eta(t):
        return t0/(t+t1)
        
    
    n = len(y) 
    M = int(n/m) # size of minibatches

    change = 0.0
    diff = tol + 1
    iter = 0
    if momentum:
        while iter < n_epochs and tol < diff: #for epoch in range(n_epochs): 
            for i in range(m): 
                # chose a minibatch ------------------
                random_index = M*np.random.randint(m)
                Xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]

                eta = time_decay_eta(iter*m+i) #time_decay_eta(epoch*m+i)
                new_change = eta * gradient(yi,Xi,beta) + delta_mom * change # add momentum
                beta -= new_change                                           # make change
                change = new_change                                          # save change

            # Calculate MSE-value----------
            y_pred_i = X @ beta
            MSE_list[iter+1] = MSE(y,y_pred_i)
            # ----------------------------

            iter += 1
            diff = np.linalg.norm(new_change)

    else:
        while iter < n_epochs and tol < diff: #for epoch in range(n_epochs): 
            for i in range(m): 
                # chose a minibatch ------------------
                random_index = M*np.random.randint(m)
                Xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]

                eta = time_decay_eta(iter*m+i) #time_decay_eta(epoch*m+i)
                new_change = eta * gradient(yi,Xi,beta)  
                beta -= new_change                                                                          
                beta -= eta * gradient(yi, Xi, beta)
            
            # Calculate MSE-value----------
            y_pred_i = X @ beta
            MSE_list[iter+1] = MSE(y,y_pred_i)
            # ----------------------------
            
            iter += 1
            diff = np.linalg.norm(new_change)
    
    if iter == n_epochs:
        iter = -1 
    
    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter, MSE_list


def sklearn_SGD(X, y, n_epochs, eta0):

    x = X[:,0].reshape(-1,1)
    x = X

    #n_epochs = Niter  # Equivalent to n_epochs in SGD()
    #eta = 0.1       # Fixed Learning Rate (Initial learning rate in tunable_eta())
    # Create the SGD Regressor model
    SGD_reg = SGDRegressor(max_iter=n_epochs, eta0=eta0)
    # Note: not momentum & fixed eta & max_iter is # epochs
    # Note: M = 1 fro SGDRegressor -> fully stochastic updates (one sample at a time)
    
    
    # Fit the model
    SGD_reg.fit(x, y.ravel())  

    # Make Prediction
    y_pred = SGD_reg.predict(x)


    return y_pred, n_epochs

def Adagrad(X,y,beta0,Niter,eta,m,delta_mom,mom_bool,SGD_bool,tol):
    beta = np.copy(beta0)

    MSE_list = np.zeros(Niter+1)
    MSE_list[0] = MSE(y, X @ beta)

    change = 0.0
    # AdaGrad parameter to avoid possible division by zero
    delta  = 1e-8   

    # ----------------- SGD - parameters ---------------
    m = int(m)
    n_epochs = int(Niter)
    n = len(y) 
    M = int(n/m) # size of minibatches
    # -------------------------------------------------
        

    
    diff  = tol + 1
    iter  = 0
    Giter = 0
    new_change = 0

    # Note: lambda = 0 
    if SGD_bool:
        while iter < n_epochs and tol < diff: #for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                random_index = M*np.random.randint(m)
                Xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]

                Giter += gradient(yi,Xi,beta)**2
                new_change = gradient(yi,Xi,beta)*eta/(delta+np.sqrt(Giter))

                beta -= new_change

            # Calculate MSE-value----------
            y_pred_i = X @ beta
            MSE_list[iter+1] = MSE(y,y_pred_i)
            # ----------------------------

            iter += 1
            diff = np.linalg.norm(new_change)

    else: # GD or momGD with AdaGrad
        while iter < Niter and tol < diff:
            Giter += gradient(y,X,beta)**2
            new_change = gradient(y,X,beta)*eta/(delta+np.sqrt(Giter))

            if mom_bool: # add momentum
                new_change += delta_mom * change
                change = new_change    
    
            beta -= new_change

            # Calculate MSE-value----------
            y_pred_i = X @ beta
            MSE_list[iter+1] = MSE(y,y_pred_i)
            # ----------------------------

            iter += 1
            diff = np.linalg.norm(new_change)



    if iter == Niter:
        iter = -1 #f"Convergence demand not met"

    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter, MSE_list

def RMSprop(X,y,beta0,n_epochs,eta,m,tol):
    beta = np.copy(beta0)

    MSE_list = np.zeros(Niter+1)
    MSE_list[0] = MSE(y, X @ beta)


    # Value for parameter rho
    rho = 0.99
    # AdaGrad parameter to avoid possible division by zero
    delta  = 1e-8   
    # ----------------- SGD - parameters ---------------
    m = int(m)
    n_epochs = int(n_epochs)
    n = len(y) 
    M = int(n/m) # size of minibatches
    # -------------------------------------------------

    diff = tol + 1
    iter = 0
    while iter < n_epochs and tol < diff: #for epoch in range(n_epochs):
        Giter = 0.0
        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            # Accumulated gradient
	        # Scaling with rho the new and the previous results
            Giter = rho * Giter + (1-rho) * gradient(yi,Xi,beta)**2
            new_change = gradient(yi,Xi,beta) * eta / (delta + np.sqrt(Giter))

            beta -= new_change
        
        # Calculate MSE-value----------
        y_pred_i = X @ beta
        MSE_list[iter+1] = MSE(y,y_pred_i)
        # ----------------------------

        iter += 1
        diff = np.linalg.norm(new_change)
    
    if iter == Niter:
        iter = -1 

    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter, MSE_list


def Adam(X,y,beta0,n_epochs,eta,m,tol):
    beta = np.copy(beta0)

    MSE_list = np.zeros(Niter+1)
    MSE_list[0] = MSE(y, X @ beta)


    iter = 0
    # Value for parameters beta1 and beta2, see https://arxiv.org/abs/1412.6980
    beta1 = 0.9
    beta2 = 0.999
    # AdaGrad parameter to avoid possible division by zero
    delta  = 1e-8   
    
    # ----------------- SGD - parameters ---------------
    m = int(m)
    n_epochs = int(n_epochs)
    n = len(y) 
    M = int(n/m) # size of minibatches
    # -------------------------------------------------

    diff = tol + 1
    iter = 0
    new_change = 0
    while iter < n_epochs and tol < diff: #for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        iter += 1
        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            first_moment = beta1 * first_moment + (1-beta1) * gradient(yi,Xi,beta)
            second_moment = beta2 * second_moment + (1-beta2) * gradient(yi,Xi,beta)**2
            first_term = first_moment / (1.0 - beta1**iter)
            second_term = second_moment / (1.0 - beta2**iter)

            new_change = eta * first_term / (np.sqrt(second_term) + delta)
            beta -= new_change

        # Calculate MSE-value----------
        y_pred_i = X @ beta
        MSE_list[iter] = MSE(y,y_pred_i)
        # ----------------------------

        #iter += 1
        diff = np.linalg.norm(new_change)
    
    if iter == Niter:
        iter = -1 

    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter, MSE_list



def plot(x, y, y_pred, title, lmd=0, Niter=0, eta=0, n_epochs=0, m=0):
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1,1,1)

    data = ax.scatter(x, y, color='r', s=15, alpha=0.5, label=r'Data')
    tilde = ax.scatter(x, y_pred, color='g', s=15, alpha=0.5, label=r'z_Pred')

    fig.suptitle(title + " Regression", fontsize=fontsize)

    #if n_epochs == 0:
    if "SGD" not in title:
        if title == "OLS":
            ax.set_title("")
        else:
            if Niter == -1: #if convegence criteria was not reached
                ax.set_title(f'Niter = Did not convergence, lmd = {lmd}, eta = {eta}', fontsize = lablesize)
            else:
                ax.set_title(f'Niter = {Niter}, lmd = {lmd}, eta = {eta}', fontsize = lablesize)
    else: # our SGD-code
        if n_epochs == -1: #if convegence criteria was not reached
            ax.set_title(f'n_epochs = Did not convergence, m = {m}\nlmd = {lmd}, eta = {eta}', fontsize = lablesize)
        else:
            ax.set_title(f'n_epochs = {n_epochs}, m = {m}\nlmd = {lmd}, eta = {eta}', fontsize = lablesize)



    
    # Add MSE-value
    ax.text(1.6, 2, f'MSE = {MSE(y, y_pred):.2e}', fontsize=lablesize-2, bbox=dict(facecolor='white', alpha=.8))
    ax.set_xlabel('x', fontsize = lablesize)
    ax.set_ylabel('y', fontsize = lablesize)
    ax.legend(loc='upper left', fontsize='small')
    ax.grid()
    fig.tight_layout()
    
    plt.savefig("Additional_Plots/" + title + ".png")
    plt.show()


def plot_heatmap(y, y_pred, title, var1, var2):
    # -----------------------Calculate MSE value ---------------
    MSE_list = np.zeros( (len(var1), len(var2)) )
    for i in range(len(var1)):
            for j in range(len(var2)):
                MSE_list[i,j] = MSE(y, y_pred[i,j,:])
    # Round variables to 2 decimal place to avoid floating-point errors
    var2 = np.round(var2, 2)
    

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(MSE_list, annot=True, cmap="coolwarm", xticklabels=var2, yticklabels=var1)

    # Set original labels and title
    if title == "GD" or title == "Momentum-GD":
        plt.xlabel(r"Learning Rate, $\eta$ []", fontsize = lablesize)
        plt.ylabel(r"L2-penalty, $\lambda$ []", fontsize = lablesize)
    else:
        plt.xlabel(r"Learning Rate, $\eta$ []", fontsize = lablesize)
        plt.ylabel(r"Number of minibatches, m []", fontsize = lablesize)

    plt.title(f"Heatmap of MSE values for {title}\n ", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("Additional_Plots/" + title + "_heatmap.png")


    # ------------------- Find idex for lowest MSE --------------
    min_val = np.min(MSE_list)
    for i in range(len(MSE_list[:,0])):
        for j in range(len(MSE_list[0,:])):
            if MSE_list[i,j] == min_val:
                i_min, j_min = i, j
    
    # -------------- Printing R2 values -------------------------
    print(f'----------------- Optimal R2 values:  {title} ------------------------')
    if title == "GD" or title == "Momentum-GD":
        print(f'R2 (lambda = {var1[i_min]}, eta = {var2[j_min]}) = {R2(y, y_pred[i_min,j_min,:]):.3e}\n')
    else:
        print(f'R2 (m = {var1[i_min]}, eta = {var2[i_min]})    = {R2(y, y_pred[i_min,j_min,:]):.3e}\n')
    
    return i_min, j_min









    

if __name__ == "__main__": # ----------------------------------------------------------------------------------------
    n = 1000
    x = 2*np.random.rand(n,1)
    y = 2 + x + 3*x**2 - 2*x**3 + 0.1*np.random.randn(n,1)

    # ------- automatic differentiation --------
    want_Autograd     = False   # (true) if want to replace alytical gradient with JAX 
    # ------------------------------------------
    want_OLS     = True
    want_GD      = True
    want_momGD   = True
    want_SGD     = True
    # --- SGD --------
    mom_SGD_bool = True   # if want (True) or not want (False) momentum
    # ----------------
    want_SGD_skl = True
    want_Adagrad = True    
    # --- Adagrad ----
    SGD_Ada_bool = True   # if want AdaGrad with SGD with momentum
    mom_Ada_bool = False  # if want momentum to AdaGrad with plain GD (works if SGD_Ada_bool = False)
    # ---------------
    want_RMSprop = True  
    want_Adam    = True   

    # Note-----------------------
    # Adagrad with GD, momGD, SGD
    # RMSprop with SGD
    # Adam    with SGD
    # ---------------------------

    if want_Autograd: 
        # Gradient wrt third argument (beta, 2 here)
        #gradient = grad(CostOLS,2)
        gradient = grad(CostOLS,2)
    else:
        gradient = grad_anl

    # Creating Design Matrix
    X = np.c_[x, x**2, x**3] # No Intecept given Scaling-process

    # Scale values
    scaler = Scaler() # Create a scaler instance

    # Scale the input data and target
    X, y = scaler.scale(X, y)

    # - Define a common random beta before
    # - So regressions functions to obtain equal beta.
    beta0 = np.random.randn(X.shape[1],y.shape[1])



    #---------------------Optimal eta ? -------------------------------
    #------------------- Will take too long time for NN ---------------
    # ------------------ This code is before NN-code ------------------
    #H = (2.0/n) * X.T @ X # Hessian Matrix 
    #EigValues, EigVectors = np.linalg.eig(H)
    # print(f"Eigenvalues of Hessian Matrix:{EigValues}")
    #eta = 1.0/np.max(EigValues) 
    #write about this would be better, but takes time for NN
    #-----------------------------------------------------------------

    # -------------------- Parameters ---------------------
    Niter = 100
    tol   = 1e-4

    eta_list = np.linspace(0, .3, 11)   
    eta_list = eta_list[1:] # remove eta = 0

    eta_list2 = np.linspace(0,0.1, 11)
    eta_list2 = eta_list2[1:]
    
    lmb_list = np.logspace(-5, 0, 6) 
    lmb_list[0] = 0 # This give OLS case (not L2-penalty)
    # lmb = 0 -> plain OLS
    # lmb > 0 -> L2 - penalty (Ridge)

    m_list = np.linspace(10, 100, 10)
    #m_list = np.linspace(10, 100, 20)

    n_epochs_list = np.linspace(10, 100, 10)
    
    

    # ----------------- Best values obtain from heatmaps -----------
    eta_best = 0.3      # obtain from GD and momGD analysis
    delta_mom = 0.2     # testing heatmap result in delta_mom in range [.2, .4] us good
    #m_best = 1          # from SGD
    #m_best_Ada = 45     # from SGD_AdaGrad
    #m_best_RMS = 49     # from SGD_RMSprop
    #m_best_A = 61       # from SGD_Adam
    #n_epochs_best = 48  # from SGD

    # PS: random_index changes for AdaGrad, RMSprop and Adam if they
    # PS: are run together. Our best values (seen in plots) is obtained 
    # PS: when running all functions together
    # ---------------------------------------------------------------

    # Note: scaler.rescale(...) rescale data
    if want_OLS:
        y_OLS = OLS(X, y) 
        y_OLS = scaler.rescale(y_OLS)

    if want_GD:
        y_GD     = np.zeros( (len(lmb_list), len(eta_list), len(y), 1) )
        Niter_GD = np.zeros( (len(lmb_list), len(eta_list)) )
        MSE_GD   = np.zeros( (len(lmb_list), len(eta_list), Niter+1) ) 
        for i in range(len(lmb_list)):
            for j in range(len(eta_list)):
                y_GD[i,j,:], Niter_GD[i,j], MSE_GD[i,j] = GD(X, y, beta0, Niter=Niter, tol=tol, eta=eta_list[j], lmb=lmb_list[i])
                y_GD[i,j,:] = scaler.rescale(y_GD[i,j])

    if want_momGD:
        y_momGD     = np.zeros( (len(lmb_list), len(eta_list), len(y), 1) )
        Niter_momGD = np.zeros( (len(lmb_list), len(eta_list)) )
        MSE_momGD   = np.zeros( (len(lmb_list), len(eta_list), Niter+1) ) 

        for i in range(len(lmb_list)):
            for j in range(len(eta_list)):
                y_momGD[i,j,:], Niter_momGD[i,j], MSE_momGD[i,j] = momGD(X, y, beta0, Niter=Niter, tol=tol, eta=eta_list[j], lmb=lmb_list[i], delta_mom=delta_mom)
                y_momGD[i,j,:] = scaler.rescale(y_momGD[i,j])
       
    if want_SGD:
        # Use best eta and lambda from GD- and momGD-analysis
        # best lambda = 0 -> OLS, no L2-penalty
        # best eta = eta0_SGD in parameter above
        
        y_SGD        = np.zeros( (len(m_list), len(eta_list), len(y), 1) )
        n_epochs_SGD = np.zeros( (len(m_list), len(eta_list)) )
        MSE_SGD      = np.zeros( (len(m_list), len(eta_list), Niter+1) ) 
        
        for i in range(len(m_list)):
            for j in range(len(eta_list)):
                y_SGD[i,j,:], n_epochs_SGD[i,j], MSE_SGD[i,j] = SGD(X, y, beta0, n_epochs=Niter, m=m_list[i], tol=tol, eta0=eta_list[j], delta_mom=delta_mom,  momentum=mom_SGD_bool)
                y_SGD[i,j,:] = scaler.rescale(y_SGD[i,j])
        

    if want_SGD_skl:
        y_SGD_skl, n_epochs_SGD_skl = sklearn_SGD(X, y, n_epochs=Niter, eta0=eta_best)
        y_SGD_skl = y_SGD_skl.reshape(-1,1) # O.G. size = (N,) -> (N,1)
        y_SGD_skl = scaler.rescale(y_SGD_skl)

    if want_Adagrad:
        if SGD_Ada_bool:
            y_Ada        = np.zeros( (len(m_list), len(eta_list2), len(y), 1) )
            n_epochs_Ada = np.zeros( (len(m_list), len(eta_list)) )
            MSE_Ada      = np.zeros( (len(m_list), len(eta_list), Niter+1) ) 
        
            for i in range(len(m_list)):
                for j in range(len(eta_list2)):
                    y_Ada[i,j,:], n_epochs_Ada[i,j], MSE_Ada[i,j] = Adagrad(X,y,beta0,Niter,eta_list2[j],m_list[i],delta_mom, mom_Ada_bool, SGD_Ada_bool, tol=tol)
                    y_Ada[i,j,:] = scaler.rescale(y_Ada[i,j])
        else:
            MSE_Ada = np.zeros( Niter+1 ) 
            y_Ada, Niter_Ada, MSE_Ada  = Adagrad(X,y,beta0,Niter,eta_best,delta_mom=delta_mom, mom_bool=mom_Ada_bool, SGD_bool=SGD_Ada_bool, tol=tol) #Adagrad(X,y,beta0,Niter,eta_best,m_best,delta_mom, mom_Ada_bool, SGD_Ada_bool, tol=tol)
            y_Ada = scaler.rescale(y_Ada)
    
    if want_RMSprop:
        y_RMS        = np.zeros( (len(m_list), len(eta_list2), len(y), 1) )
        n_epochs_RMS = np.zeros( (len(m_list), len(eta_list)) )
        MSE_RMS      = np.zeros( (len(m_list), len(eta_list), Niter+1) ) 
        
        for i in range(len(m_list)):
            for j in range(len(eta_list2)):
                y_RMS[i,j,:], n_epochs_RMS[i,j], MSE_RMS[i,j] = RMSprop(X,y,beta0, Niter, eta_list2[j],m_list[i], tol=tol)
                y_RMS[i,j,:] = scaler.rescale(y_RMS[i,j])
    
    if want_Adam:
        y_A          = np.zeros( (len(m_list), len(eta_list2), len(y), 1) )
        n_epochs_A   = np.zeros( (len(m_list), len(eta_list)) ) 
        MSE_A        = np.zeros( (len(m_list), len(eta_list), Niter+1) ) 
        
        for i in range(len(m_list)):
            for j in range(len(eta_list2)):
                y_A[i,j,:], n_epochs_A[i,j], MSE_A[i,j] = Adam(X,y,beta0, Niter, eta_list2[j],m_list[i], tol=tol)
                y_A[i,j,:] = scaler.rescale(y_A[i,j])









    # Rescale the y (data) back to the original scale
    # after using it for regression
    y = scaler.rescale(y)   


    
    

    # ---------------------------- Plotting ------------------------
    #---------- Plotting comparison ------------
    fig = plt.figure(figsize=figsize)
    fig.suptitle('MSE vs. Iterations / Epochs', fontsize=fontsize)
    ax = fig.add_subplot(1,1,1)
    n_iter = np.linspace(0,Niter,Niter+1)

    if want_OLS:
        plot(x, y, y_OLS, "OLS")

    if want_GD:
        i,j = plot_heatmap(y, y_GD, "GD", lmb_list, eta_list)
        # Plot best fit from heatplot: eta = 0.3, lambda = 0
        plot(x, y, y_GD[i,j], "GD", Niter=Niter_GD[i,j], eta=eta_list[j], lmd=lmb_list[i])
        ax.plot(n_iter, MSE_GD[i,j], label='GD')
    
    if want_momGD:
        i,j = plot_heatmap(y, y_momGD, "Momentum-GD", lmb_list, eta_list)
        # Plot best fit from heatplot: eta = 0.3, lambda = 0
        plot(x, y, y_momGD[i,j], "Momentum-GD", Niter=Niter_momGD[i,j], eta=eta_list[j], lmd=lmb_list[i])
        ax.plot(n_iter, MSE_momGD[i,j], label='mom_GD')
        
    if want_SGD:
        # Plot best fit from heatplot: m = 1, n_epochs = 48
        if mom_SGD_bool:
            #plot_heatmap(y, y_SGD, "Momentum-SGD", m_list, n_epochs_list)
            i,j = plot_heatmap(y, y_SGD, "Momentum-SGD", m_list, eta_list)
            plot(x, y, y_SGD[i,j], "Momentum-SGD", eta=eta_list[j], n_epochs=n_epochs_SGD[i,j], m=m_list[i])
            ax.plot(n_iter, MSE_SGD[i,j], label='mom_SGD')
        else:
            i,j = plot_heatmap(y, y_SGD, "SGD", m_list, eta_list)
            plot(x, y, y_SGD[i,j], "SGD", eta=eta_list[j], n_epochs=n_epochs_SGD[i,j], m=m_list[i])
            ax.plot(n_iter, MSE_SGD[i,j], label='SGD')
        
    if want_SGD_skl:
        plot(x, y, y_SGD_skl, "SGD_Scikit_Learn", n_epochs=n_epochs_SGD_skl, eta=eta_best, m=n)
    
    if want_Adagrad:
        if SGD_Ada_bool:
            i,j = plot_heatmap(y, y_Ada, "AdaGrad_SGD", m_list, eta_list2)
            plot(x, y, y_Ada[i,j], "AdaGrad_SGD", eta=eta_list2[j], n_epochs=n_epochs_Ada[i,j], m=m_list[i])
            ax.plot(n_iter, MSE_Ada[i,j], label='AdaGrad')

        elif mom_Ada_bool:
            plot(x, y, y_Ada, "AdaGrad_Momentum", Niter=Niter_Ada, eta=eta_best)
            ax.plot(n_iter, MSE_Ada, label='AdaGrad_mom_GD')
        else:
            plot(x, y, y_Ada, "AdaGrad", Niter=Niter_Ada, eta=eta_best)
            ax.plot(n_iter, MSE_Ada, label='AdaGrad_GD')
    
    if want_RMSprop:
        i,j = plot_heatmap(y, y_RMS, "RMSprop_SGD", m_list, eta_list2)
        plot(x, y, y_RMS[i,j], "RMSprop_SGD", eta=eta_list2[j], n_epochs=n_epochs_RMS[i,j], m=m_list[i])
        ax.plot(n_iter, MSE_RMS[i,j], label='RMSprop')
    if want_Adam:
        i,j = plot_heatmap(y, y_A, "Adam_SGD", m_list, eta_list2)
        plot(x, y, y_A[i,j], "Adam_SGD", eta=eta_list2[j], n_epochs=n_epochs_A[i,j], m=m_list[i])
        ax.plot(n_iter, MSE_A[i,j], label='Adam')



    
    ax.set_yscale('log') # Set the y-axis to logarithmic scale
    ax.set_ylim(8e-3,1)
    ax.set_xlabel('Iterations / Epochs []', fontsize = lablesize)
    ax.set_ylabel('MSE []', fontsize = lablesize)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid()
    fig.tight_layout()
    fig.savefig("Additional_Plots/MSE.png")
    plt.show()


    # BG-version 16.10.2024