## -------------------------------------------

import boto3
import pandas as pd
import numpy as np
from tqdm import tqdm
import miceforest as mf
from Amex_Metric import amex_metric
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import os
import sagemaker

## -------------------------------------------

## Sanity check
print('-- Process Starting --')

sess = sagemaker.Session()
s3 = boto3.resource('s3')
bucket_name = 'evan-callaghan-bucket'
bucket = s3.Bucket(bucket_name)

file_key = '/amex_train_data_balance.csv'

bucket_object = bucket.Object(file_key)
file_object = bucket_object.get()
file_content_stream = file_object.get('Body')

## Reading the data
balance = pd.read_csv(file_content_stream)

## Sanity check
print('-- Data Read -- \n')

## -------------------------------------------

## Sanity check
print('-- Data-frame imputation starting -- \n')

## Defining the input variables and dropping categorical variables
mf_balance = balance.drop(columns = ['customer_ID', 'target'])

# Building the miceforest kernel
kernel_balance = mf.ImputationKernel(mf_balance, datasets = 5, save_all_iterations = True)

## Assigning the final imputed data-frames
balance_impute = kernel_balance.complete_data(dataset = 0, inplace = False)

## Adding "customer_ID" back into the data-frames
balance_impute = pd.concat([balance[['customer_ID', 'target']], balance_impute], axis = 1)

## Sanity check
print('-- Data-frame imputation complete -- \n')

## -------------------------------------------

## Sanity check
print('-- Feature Selection Process Started --')

## Defining the input and target variables
X = balance_impute.drop(columns = ['customer_ID', 'target'])
Y = balance_impute['target']

## Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y)

## Defining the customized scoring function 
amex_function = make_scorer(amex_metric, greater_is_better = True, needs_proba = True)

## Defining empty list to store results
features_to_select = list()

## Repeating RFECV steps 10 times:
for i in tqdm(range(0, 10)):
    
    ## Runing RFECV with Random Forest as a base algorithm
    rf_rfecv = RFECV(estimator = RandomForestClassifier(n_estimators = 100, max_depth = 3), step = 5, scoring = amex_function, min_features_to_select = 3, cv = 3).fit(X_train, Y_train)
    
    ## Appending results 
    features_to_select.append(rf_rfecv.support_)
    
## Creating a data-frame to stre results
features_to_select = pd.DataFrame(features_to_select, columns = X.columns)
features_to_select = 100 * features_to_select.apply(np.sum, axis = 0) / features_to_select.shape[0]

## Producing the final output data-frame
output = pd.DataFrame(features_to_select).reset_index(drop = False)
output.columns = ['Variable', 'Selected']
output = output.sort_values(by = 'Selected', ascending = False).reset_index(drop = True)

## Exporting the resulting data-frame to the S3 Bucket
output.to_csv('balance_feature_selection.csv', index = False)

sess.upload_data(path = 'balance_feature_selection.csv', bucket = bucket_name, key_prefix = 'Kaggle-American-Express-Default')

## Sanity check
print('-- Feature selection complete -- \n')