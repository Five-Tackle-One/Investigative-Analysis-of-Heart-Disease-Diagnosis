import pandas as pd 
import numpy as np
from random import seed
from random import randrange

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



"""
Recreate the dataset with named feature values instead for display purpses

"""

def make_description_dataset(dataset,categories):    
    description_dataset = dataset.copy()[categories]
    description_dataset['sex'] = description_dataset['sex'].map({1:"Male",0:"Female"})
    description_dataset['cp']  = description_dataset['cp'].map({1:"Typical Angina",2:"Atypical Angina",3:'Non-Anginal Pain',4:'Asymptotic'})
    description_dataset['fbs']  = description_dataset['fbs'].map({1:"True",2:"False"})
    description_dataset['restecg']  = description_dataset['restecg'].map({0:"Normal",1:"Having ST-T Wave",2:"Left-Ventricular Hyerpthrophy"})
    description_dataset['exang']  = description_dataset['exang'].map({0:"No",1:"Yes"})
    description_dataset['slope']  = description_dataset['slope'].map({1:"Upsloping",2:"Flat",3:"Downsloping"})
    description_dataset['num']  = description_dataset['num'].map({1:"Presence",0:"Absence"})
    return description_dataset



def minmax_column(dataset,column):
    data = dataset.copy()
    min_ = min(data[column])
    max_ = max(data[column])
    return (data[column] - min_)/(max_ - min_)


"""
Perform Min-Max Scaling on the dataset
"""
def minmax_scale(dataset):
    features = list(dataset.iloc[:,:-1].columns)
    data = dataset.copy()
    for feature in features:
        if len(data[feature].value_counts()) > 1:
            data[feature] = minmax_column(dataset,feature)
    return data




"""
Split the dataset into train and split data

"""
def train_test_split(dataset, split=0.70):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


"""
Perform K-Fold Cross Validation and return the folded datasets

"""
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split




def add_influence(results):
    values = results.copy()
    if len(np.unique(results)) == 1:
        values[0] = 0
        values[1] = 1
    return values