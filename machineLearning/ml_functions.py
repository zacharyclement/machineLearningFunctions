# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.datasets import load_boston, load_iris, load_wine

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

boston = load_boston()
wine = load_wine()
iris = load_iris()


def Logistic_regression(data):
    X = data['data']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=5)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    c_logspace = np.logspace(-2, 2, 30)
    
    model = LogisticRegression()
    param_grid ={'C': c_logspace,
                 'penalty': ['l1', 'l2']}
    
    model_cv = GridSearchCV(model, param_grid, cv=5)
    
    model_cv.fit(X_train_scaled, y_train)
    
    y_pred = model_cv.predict(X_test_scaled)
    
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(model_cv.score(X_train_scaled, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(model_cv.score(X_test_scaled, y_test)))
    print("Tuned Model Parameter: {}".format(model_cv.best_params_))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
def SVM(data):
    X = data['data']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=5)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    c_logspace = np.logspace(-2, 2, 30)
    that_gamma = [.0001, .001, .01, .1, 1, 10]
    
    model = SVC(kernel='linear')
    param_grid ={'C': c_logspace,
                 'kernel': ['linear', 'rbf', 'poly'],
                 'gamma': that_gamma}
    
    model_cv = GridSearchCV(model, param_grid, cv=5)
    
    model_cv.fit(X_train_scaled, y_train)
    
    y_pred = model_cv.predict(X_test_scaled)
    
    print('Accuracy of support vector classifier on training set: {:.2f}'.format(model_cv.score(X_train_scaled, y_train)))
    print('Accuracy of support vector classifier on test set: {:.2f}'.format(model_cv.score(X_test_scaled, y_test)))
    print("Tuned Model Parameter: {}".format(model_cv.best_params_))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))



def Linear_regression(data):
    
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    reg = linear_model.LinearRegression()
    param_grid = 0
    cv_result = cross_val_score(reg, X, y, cv=5)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = reg.score(X_test, y_test)

    return "Linear Score: {}".format(r2)



def Lasso_regression(data):
    
    X = data['data']
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    
    param_grid = {'alpha': [.01, .1, 1, 10, 100],
                  'normalize': [True, False]}
    
    lasso = Lasso()
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
    lasso_cv.fit(X_train, y_train)
    lasso_pred = lasso_cv.predict(X_test)
    r2 = lasso_cv.score(X_test, y_test)

    return "Lasso Score: {}".format(r2), "Tuned parameters: {}".format(lasso_cv.best_params_)



def Ridge_regression(data):
    X = data['data']
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    
    param_grid = {'alpha': [.01, .1, 1, 10, 100],
                  'normalize': [True, False]}
    
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
    ridge_cv.fit(X_train, y_train)
    ridge_pred = ridge_cv.predict(X_test)
    r2 = ridge_cv.score(X_test, y_test)
    
    return "Ridge Score: {}".format(r2), "Tuned parameters: {}".format(ridge_cv.best_params_)

    





#print(Linear_regression(boston))

#print(Lasso_regression(boston))

#print(Ridge_regression(boston))

#print(Logistic_regression(wine))
    
print(SVM(wine))
