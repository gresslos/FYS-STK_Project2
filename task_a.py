import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


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


# Defining our Cost-Function
def MSE(y_data, y_model):
    n = np.size(y_data)
    return np.sum((y_data - y_model)**2) / n

# Defining Analytical Gradient of Cost-Function
# See derivations Week 39
def grad_anl(y_data, X, beta):
    n = np.size(y_data)
    return (2.0 / n)  *X.T @ (X @ beta - y_data)



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

    #print("-------------OLS Regression-------------")
    #print("Beta = \n", beta)

    return y_pred




def GD(X,y, beta0, Niter, tol, eta, lmb=0):
    beta = np.copy(beta0) # Copy to not change beta0 for nest Regression model

    diff = tol + 1
    iter = 0

    while iter < Niter and tol < diff:
        new_change = eta * (grad_anl(y, X, beta) + 2 * lmb * beta)
        beta -= new_change
        # Will be plain OLS if lmb = 0
        iter += 1
        diff = np.linalg.norm(new_change)

    if iter == Niter:
        iter = f"Did not convergence"

    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter



def momGD(X,y, beta0, Niter, tol, eta, delta_mom, lmb=0):
    beta = np.copy(beta0)

    change = 0.0
    
    diff = tol + 1
    iter = 0
    while iter < Niter and tol < diff:
        new_change = eta * grad_anl(y,X,beta) + delta_mom * change # add momentum
        beta -= new_change                                         # make change
        change = new_change                                        # save change

        iter += 1
        diff = np.linalg.norm(new_change)

    if iter == Niter:
        iter = f"Did not convergence"

    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter


def SGD(X, y, beta, n_epochs, m,  tol, eta0, delta_mom, momentum = True):
    # - Use the best learning rate (eta) from GD / momGD - analyze 
    # - as tunable_eta(0)

    n_epochs = int(n_epochs)
    m = int(m)
    beta = np.copy(beta0)

    
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
    if momentum:
        for epoch in range(n_epochs): 
            for i in range(m): 
                # chose a minibatch ------------------
                random_index = M*np.random.randint(m)
                Xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]

                eta = time_decay_eta(epoch*m+i)
                new_change = eta * grad_anl(yi,Xi,beta) + delta_mom * change # add momentum
                beta -= new_change                                           # make change
                change = new_change                                          # save change

    else:
        for epoch in range(n_epochs): 
            for i in range(m): 
                # chose a minibatch ------------------
                random_index = M*np.random.randint(m)
                Xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]

                eta = time_decay_eta(epoch*m+i)
                beta -= eta * grad_anl(yi, Xi, beta)
    
    # Make Prediction
    y_pred = X @ beta

    return y_pred, n_epochs


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

    print("-------------SGD Regression (scikit-learn)-------------")
    print("Beta = \n", SGD_reg.coef_)
    print("Intercepts = ", SGD_reg.intercept_)

    return y_pred, n_epochs


def Adagrad(X,y,beta0,Niter,n_epochs,eta,m,tol,delta_mom,mom_bool,SGD_bool):
    beta = np.copy(beta0)

    change = 0.0
    # AdaGrad parameter to avoid possible division by zero
    delta  = 1e-8   

    # ----------------- SGD - parameters ---------------
    m = int(m)
    n_epochs = int(n_epochs)
    n = len(y) 
    M = int(n/m) # size of minibatches
    # -------------------------------------------------
        

    
    diff  = tol + 1
    iter  = 0
    Giter = 0
    new_change = 0

    # Note: lambda = 0, only SGD with momentum (better then without)
    if SGD_bool:
        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                random_index = M*np.random.randint(m)
                Xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]

                Giter += grad_anl(yi,Xi,beta)**2
                new_change = grad_anl(yi,Xi,beta)*eta/(delta+np.sqrt(Giter))

                beta -= new_change

    else: # GD or momGD with AdaGrad
        while iter < Niter and tol < diff:
            Giter += grad_anl(y,X,beta)**2
            new_change = grad_anl(y,X,beta)*eta/(delta+np.sqrt(Giter))

            if mom_bool: # add momentum
                new_change += delta_mom * change
                change = new_change    
    
            beta -= new_change
            iter += 1
            diff = np.linalg.norm(new_change)


    if iter == Niter:
        iter = f"Convergence demand not met"

    # Make Prediction
    y_pred = X @ beta

    return y_pred, iter

def RMSprop(X,y,beta0,n_epochs,eta,m):
    beta = np.copy(beta0)

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
    for epoch in range(n_epochs):
        Giter = 0.0
        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            # Accumulated gradient
	        # Scaling with rho the new and the previous results
            Giter = rho * Giter + (1-rho) * grad_anl(yi,Xi,beta)**2
            new_change = grad_anl(yi,Xi,beta) * eta / (delta + np.sqrt(Giter))

            beta -= new_change

    # Make Prediction
    y_pred = X @ beta

    return y_pred, n_epochs


def Adam(X,y,beta0,n_epochs,eta,m):
    beta = np.copy(beta0)

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

    new_change = 0
    for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        iter += 1
        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            first_moment = beta1 * first_moment + (1-beta1) * grad_anl(yi,Xi,beta)
            second_moment = beta2 * second_moment+(1-beta2) * grad_anl(yi,Xi,beta)**2
            first_term = first_moment / (1.0 - beta1**iter)
            second_term = second_moment / (1.0 - beta2**iter)

            new_change = eta * first_term / (np.sqrt(second_term) + delta)
            beta -= new_change

    # Make Prediction
    y_pred = X @ beta

    return y_pred, n_epochs



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
            ax.set_title(f'Niter = {Niter}, lmd = {lmd}, eta = {eta}', fontsize = lablesize)
    elif title == "SGD_Scikit_Learn":
        ax.set_title(f'n_epochs = {n_epochs}, m = {m}\nlmd = {lmd}, eta = {eta}', fontsize = lablesize)
    else: # our SGD-code
        ax.set_title(f'n_epochs = {n_epochs}, m = {m}\nlmd = {lmd}, eta0 = {eta}', fontsize = lablesize)



    
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
    if "SGD" not in title:
        plt.xlabel(r"Learning Rate [$\eta$]", fontsize = lablesize)
        plt.ylabel(r"L2-penalty [$\lambda$]", fontsize = lablesize)
    elif title == "AdaGrad_SGD" or title == "RMSprop_SGD" or title == "Adam_SGD":
        plt.xlabel(r"Learning Rate [$\eta$]", fontsize = lablesize)
        plt.ylabel(r"Number of minibatches, m []", fontsize = lablesize)
    else:
        plt.xlabel(r"Number of epochs []", fontsize = lablesize)
        plt.ylabel(r"Number of minibatches, m []", fontsize = lablesize)

    plt.title(f"Heatmap of MSE values for {title}\n ", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("Additional_Plots/" + title + "_heatmap.png")





    

if __name__ == "__main__": # ----------------------------------------------------------------------------------------
    n = 100 + 900
    x = 2*np.random.rand(n,1)
    y = 2 + x + 2*x**2 + np.random.randn(n,1)

    """n = 100
    x = 2*np.random.rand(n,1)
    y = 4+3*x+np.random.randn(n,1)"""

    want_OLS     = False
    want_GD      = False
    want_momGD   = False
    want_SGD     = True
    want_SGD_skl = False
    want_Adagrad = True    
    want_RMSprop = True   
    want_Adam    = True   

    # Note-----------------------
    # Adagrad with GD, momGD, SGD
    # RMSprop with SGD
    # Adam    with SGD
    # ---------------------------

    # Creating Design Matrix
    X = np.c_[x, x**2] # No Intecept given Scaling-process

    # Scale values
    scaler = Scaler() # Create a scaler instance

    # Scale the input data and target
    X, y = scaler.scale(X, y)

    # - Define a common random beta before
    # - So regressions functions to obtain equal beta.
    beta0 = np.random.randn(X.shape[1],y.shape[1])



    #---------------------Optimal eta ? -------------------------------
    #------------------- Will take too long time for NN Morten said----
    #H = (2.0/n) * X.T @ X # Hessian Matrix 
    #EigValues, EigVectors = np.linalg.eig(H)
    # print(f"Eigenvalues of Hessian Matrix:{EigValues}")
    #eta = 1.0/np.max(EigValues) 
    #write about this would be better, but takes time for NN
    #-----------------------------------------------------------------

    # -------------------- Parameters ---------------------
    Niter = 100
    tol   = 1e-5

    eta_list = np.linspace(0, .3, 7)   
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
    eta_best_Ada = 0.05 # from SGD_Adagrad (and for RMSprop & Adam
    lambda_best = 0     # obtain from GD and momGD analysis
    delta_mom = 0.2     # testing heatmap result in delta_mom in [.2, .4] good
    m_best = 1          # from SGD
    m_best_Ada = 45     # from SGD_AdaGrad
    m_best_RMS = 49     # from SGD_RMSprop
    m_best_A = 61       # from SGD_Adam
    n_epochs_best = 48  # from SGD

    # PS: random_index changes for AdaGrad, RMSprop and Adam if they
    # PS: are run together. Our best values is obtained for these 
    # PS: functions run together
    # ---------------------------------------------------------------

    # Note: scaler.rescale(...) rescale data
    if want_OLS:
        y_OLS = OLS(X, y) 
        y_OLS = scaler.rescale(y_OLS)

    if want_GD:
        y_GD = np.zeros( (len(lmb_list), len(eta_list), len(y), 1) )
        for i in range(len(lmb_list)):
            for j in range(len(eta_list)):
                y_GD[i,j,:], Niter_GD = GD(X, y, beta0, Niter=Niter, tol=tol, eta=eta_list[j], lmb=lmb_list[i])
                y_GD[i,j,:] = scaler.rescale(y_GD[i,j])

    if want_momGD:
        y_momGD = np.zeros( (len(lmb_list), len(eta_list), len(y), 1) )

        for i in range(len(lmb_list)):
            for j in range(len(eta_list)):
                y_momGD[i,j,:], Niter_momGD = momGD(X, y, beta0, Niter=Niter, tol=tol, eta=eta_list[j], lmb=lmb_list[i], delta_mom=delta_mom)
                y_momGD[i,j,:] = scaler.rescale(y_momGD[i,j])
       
    if want_SGD:
        # Use best eta and lambda from GD- and momGD-analysis
        # best lambda = 0 -> OLS, no L2-penalty
        # best eta = eta0_SGD in parameter above
        mom_SGD_bool = False # change if want (True) or not want (False) momentum

        """y_SGD = np.zeros( (len(m_list), len(n_epochs_list), len(y), 1) )
        
        for i in range(len(m_list)):
            for j in range(len(n_epochs_list)):
                y_SGD[i,j,:], n_epochs_SGD = SGD(X, y, beta0, n_epochs=n_epochs_list[j], m=m_list[i], tol=tol, eta0=eta_best, delta_mom=delta_mom,  momentum=mom_SGD_bool)
                y_SGD[i,j,:] = scaler.rescale(y_SGD[i,j])"""
        
        y_SGD = np.zeros( (len(m_list), len(eta_list2), len(y), 1) )
        
        for i in range(len(m_list)):
            for j in range(len(eta_list2)):
                y_SGD[i,j,:], n_epochs_SGD = SGD(X, y, beta0, n_epochs=Niter, m=m_list[i], tol=tol, eta0=eta_list2[j], delta_mom=delta_mom,  momentum=mom_SGD_bool)
                y_SGD[i,j,:] = scaler.rescale(y_SGD[i,j])
        

    if want_SGD_skl:
        n_epochs_SGD_skl = 10
        y_SGD_skl, n_epochs_SGD_skl = sklearn_SGD(X, y, n_epochs=n_epochs_SGD_skl, eta0=eta_best)
        y_SGD_skl = y_SGD_skl.reshape(-1,1) # O.G. size = (N,) -> (N,1)
        y_SGD_skl = scaler.rescale(y_SGD_skl)

    if want_Adagrad:
        SGD_Ada_bool = True # if want AdaGrad with SGD with momentum
        mom_Ada_bool = False  # if want momentum to AdaGrad with plain GD (works if SGD_Ada_bool = False)

        if SGD_Ada_bool:
            m_Ada_list = np.linspace(1,65,17)
            m_Ada_list = m_list
            y_Ada = np.zeros( (len(m_Ada_list), len(eta_list2), len(y), 1) )
        
            for i in range(len(m_Ada_list)):
                for j in range(len(eta_list2)):
                    y_Ada[i,j,:], n_epochs_Ada = Adagrad(X,y,beta0,Niter,n_epochs_best,eta_list2[j],m_Ada_list[i],tol,delta_mom, mom_Ada_bool, SGD_Ada_bool)
                    y_Ada[i,j,:] = scaler.rescale(y_Ada[i,j])
        else:
            y_Ada, Niter_Ada = Adagrad(X,y,beta0,Niter,n_epochs_best,eta_best,m_best,tol,delta_mom, mom_Ada_bool, SGD_Ada_bool)
            y_Ada = scaler.rescale(y_Ada)
    
    if want_RMSprop:
        m_RMS_list = np.linspace(1,65,17)
        m_RMS_list = m_list
        y_RMS = np.zeros( (len(m_RMS_list), len(eta_list2), len(y), 1) )
        
        for i in range(len(m_RMS_list)):
            for j in range(len(eta_list2)):
                y_RMS[i,j,:], n_epochs_RMS = RMSprop(X,y,beta0,n_epochs_best,eta_list2[j],m_RMS_list[i])
                y_RMS[i,j,:] = scaler.rescale(y_RMS[i,j])
    
    if want_Adam:
        m_A_list = np.linspace(1,65,17)
        m_A_list = m_list
        y_A = np.zeros( (len(m_A_list), len(eta_list2), len(y), 1) )
        
        for i in range(len(m_A_list)):
            for j in range(len(eta_list2)):
                y_A[i,j,:], n_epochs_A = Adam(X,y,beta0,n_epochs_best,eta_list2[j],m_A_list[i])
                y_A[i,j,:] = scaler.rescale(y_A[i,j])









    # Rescale the y (data) back to the original scale
    # after using it for regression
    y = scaler.rescale(y)   


   

    # ---------------------------- Plotting ------------------------
    if want_OLS:
        plot(x, y, y_OLS, "OLS")

    if want_GD:
        plot_heatmap(y, y_GD, "GD", lmb_list, eta_list)
        # Plot best fit from heatplot: eta = 0.3, lambda = 0
        plot(x, y, y_GD[0,-1,:], "GD", Niter=Niter_GD, eta=eta_best)
        
    if want_momGD:
        plot_heatmap(y, y_momGD, "Momentum-GD", lmb_list, eta_list)
        # Plot best fit from heatplot: eta = 0.3, lambda = 0
        plot(x, y, y_momGD[0,-1,:], "Momentum-GD", Niter=Niter_momGD, eta=eta_best)
        
    if want_SGD:
        # Plot best fit from heatplot: m = 1, n_epochs = 48
        if mom_SGD_bool:
            #plot_heatmap(y, y_SGD, "Momentum-SGD", m_list, n_epochs_list)
            plot_heatmap(y, y_SGD, "Momentum-SGD", m_list, eta_list2)
            plot(x, y, y_SGD[0,1], "Momentum-SGD", eta=eta_best, n_epochs=n_epochs_best, m=m_best)
        else:
            plot_heatmap(y, y_SGD, "SGD", m_list, n_epochs_list)
            plot(x, y, y_SGD[0,1], "SGD", eta=eta_best, n_epochs=n_epochs_best, m=m_best)
        
    if want_SGD_skl:
        plot(x, y, y_SGD_skl, "SGD_Scikit_Learn", n_epochs=n_epochs_SGD_skl, eta=eta_best, m=n)
    
    if want_Adagrad:
        if SGD_Ada_bool:
            #plot_heatmap(y, y_Ada, "AdaGrad_SGD", m_Ada_list, eta_list)
            plot_heatmap(y, y_Ada, "AdaGrad_SGD", m_Ada_list, eta_list2)
            plot(x, y, y_Ada[11,0], "AdaGrad_SGD", eta=eta_best_Ada, n_epochs=n_epochs_best, m=m_best_Ada)
            # i = 11 -> m = 45
        elif mom_Ada_bool:
            plot(x, y, y_Ada, "AdaGrad_Momentum", Niter=Niter_Ada, eta=eta_best)
        else:
            plot(x, y, y_Ada, "AdaGrad", Niter=Niter_Ada, eta=eta_best)
    
    if want_RMSprop:
        #plot_heatmap(y, y_RMS, "RMSprop_SGD", m_RMS_list, eta_list)
        plot_heatmap(y, y_Ada, "AdaGrad_SGD", m_Ada_list, eta_list2)
        plot(x, y, y_RMS[12,0], "RMSprop_SGD", eta=eta_best_Ada, n_epochs=n_epochs_best, m=m_best_RMS)
        # i = 12 -> m = 49
    if want_Adam:
        #plot_heatmap(y, y_A, "Adam_SGD", m_A_list, eta_list)
        plot_heatmap(y, y_Ada, "AdaGrad_SGD", m_Ada_list, eta_list2)
        plot(x, y, y_A[-2,0], "Adam_SGD", eta=eta_best_Ada, n_epochs=n_epochs_best, m=m_best_A)
        # i = -2 -> m = 61



    
    