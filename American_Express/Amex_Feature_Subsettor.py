## -------------------------------------------

import boto3
import pandas as pd
import numpy as np
import miceforest as mf

import os
import sagemaker

## -------------------------------------------

## Sanity check
print('-- Process Starting --')

sess = sagemaker.Session()
s3 = boto3.resource('s3')
bucket_name = 'evan-callaghan-bucket'
bucket = s3.Bucket(bucket_name)

file_key = 'Kaggle-American-Express-Default/amex_train_data_balance.csv'
file_key2 = 'Kaggle-American-Express-Default/amex_test_data.csv'

bucket_object = bucket.Object(file_key)
bucket_object2 = bucket.Object(file_key2)

file_object = bucket_object.get()
file_object2 = bucket_object2.get()

file_content_stream = file_object.get('Body')
file_content_stream2 = file_object2.get('Body')

## Reading the data
train = pd.read_csv(file_content_stream)
test = pd.read_csv(file_content_stream2)

## Sanity check
print('-- Data Read -- \n')

## -------------------------------------------

## Sanity check
print('-- Data-frame imputation starting -- \n')

## Defining the input variables and dropping categorical variables
mf_data = data.drop(columns = ['customer_ID', 'target'])

# Building the miceforest kernel
kernel_data = mf.ImputationKernel(mf_data, datasets = 5, save_all_iterations = True)

## Assigning the final imputed data-frames
data_impute = kernel_data.complete_data(dataset = 0, inplace = False)

## Adding "customer_ID" back into the data-frames
data_impute = pd.concat([data[['customer_ID', 'target']], data_impute], axis = 1)

## Sanity check
print('-- Data-frame imputation complete -- \n')

## -------------------------------------------

## Subsetting the data to only include the most important features
data = data_impute[['customer_ID', 'B_1', 'B_2', 'B_3', 'B_4', 'B_5']]

## Sanity check
print('-- Data subsetting complete -- \n')

## -------------------------------------------

## Exporting the resulting training data-frame to the S3 Bucket
data.to_csv('amex_train_data_balance_final.csv', index = False)

sess.upload_data(path = 'amex_train_data_balance_final.csv', bucket = bucket_name, key_prefix = 'Kaggle-American-Express-Default')

## Sanity check
print('-- Complete -- \n')