## pip install miceforest

## -------------------------------------------

import boto3
import pandas as pd
import numpy as np
import miceforest as mf

import os
import sagemaker

## -------------------------------------------

## Read test data-frame
test = ...

## -------------------------------------------

## Sanity check
print('-- Test data-frame imputation starting -- \n')

## Defining the input variables and dropping categorical variables
mf_test = test.drop(columns = ['customer_ID'])

# Building the miceforest kernel
kernel_test = mf.ImputationKernel(mf_test, datasets = 5, save_all_iterations = True)

## Assigning the final imputed data-frames
test_impute = kernel_test.complete_data(dataset = 0, inplace = False)

## Adding "customer_ID" back into the data-frames
test_impute = pd.concat([test[['customer_ID']], test_impute], axis = 1)

## Sanity check
print('-- Test data-frame imputation complete -- \n')

## -------------------------------------------

## Sanity check
print('-- Starting feature aggregations on the test data-frame -- \n')

## Creating aggregation functions
def data_range(x):
    return x.max() - x.min()

def correlation(x):
    return pd.Series(x.values).corr(other = pd.Series(x.index), method = 'pearson')

## Creating new Payment features with the cleaned test data-frame
aggregated_vars_test = test_impute.groupby('customer_ID').agg({'B_1':['median'], 'B_2':['mean', 'sum'], 'B_3':[data_range], 'B_4':[correlation], 'B_9':['mean', 'median', 'sum'], 'B_18':['sum'], 'B_26':['mean']}).reset_index(drop = False)

## Renaming variables
aggregated_vars_test.columns = ['customer_ID', 'B_1_median', 'B_2_mean', 'B_2_sum', 'B_3_data_range', 'B_4_correlation', 'B_9_mean', 'B_9_median', 'B_9_sum', 'B_18_sum', 'B_26_mean']

## Sanity check
print('-- Testing aggregations data-frame complete -- \n')

## -------------------------------------------

## Sanity check
print('-- Final test data-frame imputation starting -- \n')

## Defining the input variables and dropping categorical variables
mf_data = aggregated_vars_test.drop(columns = ['customer_ID'])

# Building the miceforest kernel
kernel_data = mf.ImputationKernel(mf_data, datasets = 5, save_all_iterations = True)

## Assigning the final imputed data-frames
data_impute = kernel_data.complete_data(dataset = 0, inplace = False)

## Adding "customer_ID" back into the data-frames
data_impute = pd.concat([aggregated_vars_test[['customer_ID']], data_impute], axis = 1)

## Sanity check
print('-- Final test data-frame imputation complete -- \n')

## -------------------------------------------

## Exporting the resulting training data-frame as a csv file
data_impute.to_csv('amex_test_data_balance_final.csv', index = False)
