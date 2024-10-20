import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score
from sklearn.model_selection import  train_test_split 

"""
copied from "Exercises Week 42: Logistic Regression and Optimization, reminders 
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
        
        #boolean to change which version of sigmoid to use
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
            gradient = (X.T @ (y_predicted - y))/n_data
            
            # Update beta_logreg
            self.beta_logreg -= self.learning_rate*gradient
    
    
    #this is the one function that is actually made from scratch
    def SGDfit(self, X, y):
        
        n_data, num_features = X.shape
        self.beta_logreg = np.zeros(num_features)
        M = int(n_data/self.m)
        
        t0 = self.t1 * self.learning_rate
        
        change = 0.0
        for i in range(self.n_epochs):
            for j in range(self.m):
                random_index = M*np.random.randint(self.m)
                Xi = X[random_index:random_index+M, :]
                yi = y[random_index:random_index+M]
                
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
        
        return [1 if i >= 0.5 else 0 for i in y_predicted]



if __name__ == "__main__":
    # Sample data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    model = LogisticRegression(learning_rate=0.001, t1=10, gamma=0.1, lam=1e-2,
                               num_iterations=1000, n_epochs=10000, 
                               m=8, overflow=True
                               )
    
    model.SGDfit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Test set accuracy with Logistic Regression: = {accuracy_score(y_test, predictions):.3f}")
    print(f"log_loss = {log_loss(y_test, predictions)}")
    print(f"Confusion Matrix = {confusion_matrix(y_test, predictions, normalize='true')}")
    print('\n')
    
"""
20.10.2024, output:
Test set accuracy with Logistic Regression: = 0.930
log_loss = 2.5205352020361644
Confusion Matrix = [[0.8490566  0.1509434 ]
[0.02222222 0.97777778]]

If I have understood this correctly:
Only missed 2% of the people who actually has cancer but 15% of healthy patients
got really scared.
Our model is "trigger happy".
When you are looking for cancer this isn't the worst quality but of course more
accuracy is better.
"""



import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split 
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# Logistic Regression
logreg = LogisticRegression(solver='lbfgs') #L2 penalty is default in LogisticRegression
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
#Cross validation
accuracy = cross_validate(logreg, X_test, y_test, cv=10)['test_score']
print(f"Cross validation test scores: {accuracy}")
print("Test set accuracy with Logistic Regression (Scikit-Learn): {:.3f}".format(logreg.score(X_test, y_test)))
print(f"log_loss = {log_loss(y_test, predictions)}")
print(f"Confusion Matrix = {confusion_matrix(y_test, predictions, normalize='true')}")