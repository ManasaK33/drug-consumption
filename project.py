# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

def drop_variables(data):
    
    zero_cols = []
    non_unique_cols = []
    null_cols = []
    
    # check for variables to remove
    for column in data:
        # check for a redundant variable
        n_unique = len(data[column].value_counts())
        if n_unique == 1:
            non_unique_cols.append((column, n_unique))
        # check for 0 values
        n_zeros = np.count_nonzero(data[column].values == 0)
        if n_zeros > 0:
            zero_cols.append((column, n_zeros))
        # check for n/a values
        n_null = np.count_nonzero(data[column].isnull() == True)
        if n_null > 0:
            null_cols.append((column, n_null))
    
    # drop variables with 0 values
    for (column, count) in zero_cols:
        percentage = count/len(data.index) * 100.0
        # drop variable where 30% of values are 0
        if percentage >= 30.0:
            print('dropping %-10s # zero values: %d'%(column+',', count))
            del data[column]
            
    # drop redundant variables
    for (column, _) in non_unique_cols:
        print('dropping %s'%(column))
        del data[column]
    
    # drop missing variables
    for (column, count) in null_cols:
        percentage = count/len(data.index)
        # drop variable where 30% of values are missing
        if percentage >= 30.0:
            print('dropping %-10s # missing values: %d'%(column+',', count))
            del data[column]
    

data = pd.read_csv('drug_consumption.csv')

# consider numpy.inf as n/a value
pd.options.mode.use_inf_as_na = True

# label encode categorical variables
for column in data.loc[:,'Alcohol':'VSA']:
    # get label encoding for column
    data[column] = data[column].astype('category').cat.codes
    # convert column to numeric type
    data[column] = data[column].astype('int32')

# drop fake drug
del data['Semer'] 
# drop ID variable
del data['ID']




print(data.info())


