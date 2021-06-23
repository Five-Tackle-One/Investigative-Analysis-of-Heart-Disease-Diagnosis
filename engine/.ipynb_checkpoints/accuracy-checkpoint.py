import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import process_time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from . import data_preparation as dp




def sigmoid(z):
    return 1 / (1 + np.exp(-z))

"""
Given the dataset and the coefficients return a list of probabilities
"""
def predict(dataset,coef):
    data = dataset.iloc[:,:-1].copy()
    results = []
    for data_point in range(len(data)):
        z = coef.T@data.iloc[data_point]
        results.append(sigmoid(z)[0])
    return results


"""
Given the dataset, and the coefficients perform predictions and threshold the results to zero or one
"""
def thresold_results(data,coef):
    threshold = lambda x: 1 if x >= 0.5 else 0
    predictions = predict(data,coef)
    results = [threshold(prediction) for prediction in predictions]
    return results



"""
Given the actual and predicted values, return the accuracy

"""

def accuracy_metric(actual, predicted):
    accuracy = 0
    for acc,pred in zip(actual,predicted):
        if acc == pred:
            accuracy += 1
    return (accuracy/len(actual)) * 100


def display_confusion_matrices(y_test,results,type_titles):
    fig1, f1_axes = plt.subplots(ncols=len(results), nrows=1, constrained_layout=True,figsize=(15,6))
    for result in range(len(results)):
        conf_data = confusion_matrix(y_test,results[result])
        df_cm = pd.DataFrame(conf_data, columns=np.unique(y_test), index = np.unique(dp.add_influence(results[result])))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16},fmt='g',ax=f1_axes[result])# font si
        f1_axes[result].set_title(type_titles[result])

"""
Display the confusion matrix

"""

def display_confusion_matrix(y_test,results):
    conf_data = confusion_matrix(y_test,results)
    df_cm = pd.DataFrame(conf_data, columns=np.unique(y_test), index = np.unique(dp.add_influence(results)))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16},fmt='g')# font si
    
    
    
"""
Plot the ROC Curve
"""
def plot_roc_curve(y_test,predictions):
    ns_probs = [0 for _ in range(len(y_test))] # fit a model
    lr_probs = predictions # keep probabilities for the positive outcome only
    ns_auc = roc_auc_score(y_test, ns_probs) # calculate scores
    lr_auc = roc_auc_score(y_test, lr_probs)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs) # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill') # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    
    
def plot_cost_functions(nepochs,errors,type_titles):
    colors = ['b','r','g']
    fig1, f1_axes = plt.subplots(ncols=len(errors), nrows=1, constrained_layout=True,figsize=(15,6))
    for error in range(len(errors)):
        f1_axes[error].plot(range(nepochs)[10:],errors[error][10:],color=colors[error],label=type_titles[error])
        f1_axes[error].set_title("Cost Function Against The Number of Iterations")
        f1_axes[error].set_ylabel("Cost")
        f1_axes[error].set_xlabel("Number of Iterations")
        f1_axes[error].legend()
"""
Plot the ROC Curve
"""
def plot_roc_curves(y_test,predictions,type_titles):
    fig1, f1_axes = plt.subplots(ncols=len(predictions), nrows=1, constrained_layout=True,figsize=(15,6))
    for prediction in range(len(predictions)):
        ns_probs = [0 for _ in range(len(y_test))] # fit a model
        lr_probs = predictions[prediction] # keep probabilities for the positive outcome only
        ns_auc = roc_auc_score(y_test, ns_probs) # calculate scores
        lr_auc = roc_auc_score(y_test, lr_probs)
        print('No Skill for {}: ROC AUC={}'.format(type_titles[prediction],ns_auc*100))
        print('{}: ROC AUC={}'.format(type_titles[prediction],lr_auc*100))
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs) # calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        f1_axes[prediction].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill') # plot the roc curve for the model
        f1_axes[prediction].plot(lr_fpr, lr_tpr, marker='.', label=type_titles[prediction])
        f1_axes[prediction].set_xlabel('False Positive Rate')
        f1_axes[prediction].set_ylabel('True Positive Rate')
        f1_axes[prediction].legend()