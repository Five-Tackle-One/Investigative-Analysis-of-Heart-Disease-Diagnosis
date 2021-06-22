import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import process_time
from sklearn.linear_model import LinearRegression
from . import accuracy as acc


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    

def sgd(dataset,l_rate=0.3,n_epoch=1):
    data = dataset.copy()
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    coefficients = np.zeros(len(X.columns))
    sum_error = []
    for epoch in range(n_epoch):
        for observation in range(len(data)):
            yhat = sigmoid(coefficients.T@X.iloc[observation])
            error = Y.iloc[observation] - yhat
            coefficients[0] = coefficients[0] + l_rate * error * yhat *(1.0 - yhat)
            for i in range(1,len(coefficients)):
                coefficients[i] = coefficients[i] + l_rate * error * yhat * (1.0 - yhat) * X.iloc[observation,i]
        sum_error.append(error**2)
    return coefficients,sum_error



def validation(dataset,testing_data,y_test,learning_rates =  [0.2,0.3,0.4,0.5],epochs = [50,100,150,200]):
    data = dataset.copy()
    test = testing_data.copy()
    accuracies = []
    for learning_rate,epoch in zip(learning_rates,epochs):
        coefficients,_ = sgd(data,learning_rate,epoch)
        coefficients = np.array(coefficients).reshape(len(coefficients),1)
        predictions = acc.predict(test,coefficients)
        results = acc.thresold_results(test,coefficients)
        accuracies.append(acc.accuracy_metric(y_test,np.array(results)))
    return np.argmax(np.array(accuracies))