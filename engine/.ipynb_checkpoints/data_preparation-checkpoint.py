import pandas as pd 
import numpy as np

"""
The previous researchers were distinguishing the present of a heart disease vs the absence of heart disease.
However, values (1,2,3,4) were regarded as present while (0) was absence.
We,instead, will make every value  > 0 just 1

Input
dataset: dataset 
-----------
Return dataset with all values above 0 being 1
"""
def group_target(dataset,target='num'):
    data = dataset.copy()
    if len(data[target].value_counts()) > 2:
        unique_indices = list(sorted(data[target].value_counts().index)[1:])
        data = data.replace(unique_indices,1)
    return data




"""
Given a dataset, add an array on ones in the first column
Input
dataset:dataset 
---------------------
Return a dataset with a bias column in the first column
"""
def add_bias(dataset):
    data = dataset.copy()
    bias = np.ones(len(data))
    data.insert(0,"Bias",bias)
    return data