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




"""
Encode the column 
"""
def encode_column(dataset,column):
    data = dataset.copy()
    data = data.join(pd.get_dummies(data[column],prefix=column))
    data.drop(columns=[column],inplace=True)
    return data



"""
Perform one-hot encoding on a dataset
"""
def one_hot_encoding(dataset,dummy_labels = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca'],target='num'):
    data = dataset.copy()
    target_values = data[target]
    for dummy_label in dummy_labels:
        data = encode_column(data,dummy_label)
    data.drop(columns=[target],inplace=True)
    data[target] = target_values
    return data


def one_hot_encoding_2(dataset,dummy_labels = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca'],target='num'):
    data = dataset.copy()
    encode_data = pd.get_dummies(data, columns=dummy_labels, prefix=dummy_labels)
    target_values = encode_data[target]
    encode_data = encode_data.drop(columns=target)
    encode_data[target] = target_values
    return encode_data


"""
Given a dataset return a list of features that have missing values
"""
def get_missing_features(dataset):
    data = dataset.copy()
    data_missing_dict_sum  = dict(data.isnull().sum())
    data_missing_dict = dict(filter(lambda elem: elem[1] > 0, data_missing_dict_sum.items()))
    missing_features = list(data_missing_dict.keys())
    return missing_features

"""
Given a dataset 
1: Find the missing values 
2: Replace the missing values with the mode
"""
def replace_missing_with_mode(dataset):
    data = dataset.copy()
    missing_features = get_missing_features(data)
    for missing_feature in missing_features:
        mode = data[missing_feature].mode().values[0]
        data[missing_feature] = data[missing_feature].fillna(mode)
    return data