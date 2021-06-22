import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def count_plot_categories(description_dataset,categories,names,n=3,m=2):
    fig1, f1_axes = plt.subplots(ncols=n, nrows=m, constrained_layout=True,figsize=(15,10))
    for i in range(m):
        for j in range(n):
            sns.countplot(data=description_dataset, x=categories[i*m + j],ax=f1_axes[i,j])
            f1_axes[i,j].set_xlabel(names[categories[i*m + j]])
    fig1.show()
    
    
    
def box_plot_continuous(drop_missing_dataset,continuous,names,n=3,m=2,hue='num'):
    fig1, f1_axes = plt.subplots(ncols=n, nrows=m, constrained_layout=True,figsize=(15,10))
    for i in range(m):
        for j in range(n):
            sns.boxplot(data=drop_missing_dataset, x=continuous[i*m + j], hue=hue,ax=f1_axes[i,j])
            f1_axes[i,j].set_xlabel(names[continuous[i*m + j]])
    fig1.show()
    
    
def hist_plot_continuous(drop_missing_dataset,continuous,names,n=3,m=2,hue='num'):
    fig1, f1_axes = plt.subplots(ncols=n, nrows=m, constrained_layout=True,figsize=(15,10))
    for i in range(m):
        for j in range(n):
            sns.histplot(data=drop_missing_dataset, x=continuous[i*m + j], hue=hue,ax=f1_axes[i,j])
            f1_axes[i,j].set_xlabel(names[continuous[i*m + j]])