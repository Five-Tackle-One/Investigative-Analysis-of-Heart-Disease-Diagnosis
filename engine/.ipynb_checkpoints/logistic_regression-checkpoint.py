import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import process_time
from sklearn.linear_model import LinearRegression
from . import accuracy as acc
from math import log

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

def sgd_mini_batch(dataset,l_rate=0.4,n_epoch=150,batch_size=32):
    data = dataset.copy()
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    coefficients = np.zeros(len(X.columns))
    sum_error = []
    for epoch in range(n_epoch):
        mini_batches = create_minibatches(X,Y,batch_size)
        for mini_batch in mini_batches:
             for observation in range(len(mini_batch)):
                X_batch = mini_batch.iloc[:,:-1]
                y_batch = mini_batch.iloc[:,-1]
                yhat = sigmoid(coefficients.T@X_batch.iloc[observation])
                error = -y_batch.iloc[observation]*log(yhat) - (1-y_batch.iloc[observation])*log(1-yhat)
                diff = y_batch.iloc[observation] - yhat
                coefficients[0] = coefficients[0] + l_rate * error * yhat *(1.0 - yhat)
                for i in range(1,len(coefficients)):
                    coefficients[i] = coefficients[i] + l_rate * error * yhat * (1.0 - yhat) * X.iloc[observation,i]
        sum_error.append(error)
    return coefficients,sum_error


def sgd_bincross(dataset,l_rate=0.3,n_epoch=1):
    data = dataset.copy()
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    coefficients = np.zeros(len(X.columns))
    sum_error = []
    for epoch in range(n_epoch):
        for observation in range(len(data)):
            yhat = sigmoid(coefficients.T@X.iloc[observation])
            error = -Y.iloc[observation]*log(yhat) - (1-Y.iloc[observation])*log(1-yhat)
            diff = Y.iloc[observation] - yhat
            coefficients[0] = coefficients[0] + l_rate * diff
            for i in range(1,len(coefficients)):
                coefficients[i] = coefficients[i] + l_rate * diff * X.iloc[observation,i]
        sum_error.append(error)
    return coefficients,sum_error

def sgd_bincross_with_mini_batch(dataset,l_rate=0.4,n_epoch=150,batch_size=32):
    data = dataset.copy()
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    coefficients = np.zeros(len(X.columns))
    sum_error = []
    sums = []
    for epoch in range(n_epoch):
        mini_batches = create_minibatches(X,Y,batch_size)
        for mini_batch in mini_batches:
            error = 0
            for observation in range(len(mini_batch)):
                X_batch = mini_batch.iloc[:,:-1]
                y_batch = mini_batch.iloc[:,-1]
                yhat = sigmoid(coefficients.T@X_batch.iloc[observation])
                error = -y_batch.iloc[observation]*log(yhat) - (1-y_batch.iloc[observation])*log(1-yhat)
                diff = y_batch.iloc[observation] - yhat
                coefficients[0] = coefficients[0] + l_rate * diff
                for i in range(1,len(coefficients)):
                    coefficients[i] = coefficients[i] + l_rate * diff * X_batch.iloc[observation,i]
            sum_error.append(error)
        sums.append(min(sum_error))
    return coefficients,sums

def sgd_with_regularization(dataset,l_rate=0.3,n_epoch=1,regularized_term=0.5):
    data = dataset.copy()
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    coefficients = np.zeros(len(X.columns))
    sum_error = []
    for epoch in range(n_epoch):
        for observation in range(len(data)):
            yhat = sigmoid(coefficients.T@X.iloc[observation])
            error = -Y.iloc[observation]*log(yhat) - (1-Y.iloc[observation])*log(1-yhat)
            diff = Y.iloc[observation] - yhat
            coefficients[0] = coefficients[0] + l_rate * diff * yhat *(1.0 - yhat)
            for i in range(1,len(coefficients)):
                coefficients[i] = coefficients[i] + l_rate * (diff * yhat * (1.0 - yhat) * X.iloc[observation,i] - (regularized_term/len(data))*coefficients[i])
#                 coefficients[i] = coefficients[i] + l_rate * (diff * X.iloc[observation,i] - (regularized_term/len(data))*coefficients[i])
        sum_error.append(error-regularized_term * sum(coefficients**2))
    return coefficients,sum_error


def average_results(training_data,testing_data,y_test,lrate=0.4,nepochs=150,regularized_term=0.1,niters=5,batch_size=32,type_="normal"):
    coefficients=[];_=[];results=[];accuracies = []
    for i in range(niters):
        if type_ == "normal":
            coefficients,_ = sgd(training_data,lrate,nepochs) # lrate = 0.4, npochs = 150
        elif type_ == "bincross":
            coefficients,_ = sgd_bincross(training_data,lrate,nepochs) # lrate = 0.4, npochs = 150
        elif type_ == "regularization":
            coefficients,_ = sgd_with_regularization(training_data,lrate,nepochs,regularized_term) # lrate = 0.4, npochs = 150
        elif type_ == "bincross_mini_batch":
            coefficients,_ = sgd_bincross_with_mini_batch(training_data,lrate,nepochs,batch_size) # lrate = 0.4, npochs = 150
        elif type_ == "mini_batch":
            coefficients,_ = sgd_mini_batch(training_data,lrate,nepochs,batch_size) # lrate = 0.4, npochs = 150
        coefficients = np.array(coefficients).reshape(len(coefficients),1)
        predictions = acc.predict(testing_data,coefficients)
        results = acc.thresold_results(testing_data,coefficients)
        accuracies.append(acc.accuracy_metric(y_test,np.array(results)))
    avg_acc = sum(accuracies)/len(accuracies)
    return avg_acc,predictions,results,_

"""
Given hyperparameters, perform hyperparameter tuning by trying out different predetermined values and select the one that after trainin against validation data gives the highest accuracy

This method does not actually see the testing data

"""
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


def validation_with_regularization(dataset,testing_data,y_test,learning_rates =  [0.2,0.3,0.4,0.5],epochs = [50,100,150,200],regularization_terms=[0,0.005,0.008,0.1]):
    data = dataset.copy()
    test = testing_data.copy()
    accuracies = []
    for learning_rate,epoch,regularization_term in zip(learning_rates,epochs,regularization_terms):
        coefficients,_ = sgd_with_regularization(data,learning_rate,epoch,regularization_term)
        coefficients = np.array(coefficients).reshape(len(coefficients),1)
        predictions = acc.predict(test,coefficients)
        results = acc.thresold_results(test,coefficients)
        accuracies.append(acc.accuracy_metric(y_test,np.array(results)))
    return np.argmax(np.array(accuracies))

"""
Create mini batches for mini-batch gradient descent
"""
def create_minibatches(X,y,batch_size=32):
    data = X.join(y)
    n_minibatches = data.shape[0] // batch_size
    data = data.sample(frac=1)
    mini_batches = []
    i = 0
    for i in range(n_minibatches + 1):
        mini_batch = data.iloc[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch.iloc[:, :-1]
        Y_mini = mini_batch.iloc[:, -1]
        mini_batches.append(X_mini.join(Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data.iloc[i * batch_size:data.shape[0]]
        X_mini = mini_batch.iloc[:, :-1]
        Y_mini = mini_batch.iloc[:, -1]
    mini_batches.append(X_mini.join(Y_mini))
    return mini_batches