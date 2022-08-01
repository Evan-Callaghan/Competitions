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

## Creating data-type dictionary for reading the test data-frame
dtype_dict = {'customer_ID': "object", 'S_2': "object", 'P_2': 'float16', 'D_39': 'float16', 'B_1': 'float16','B_2': 'float16', 'R_1': 'float16','S_3': 'float16','D_41': 'float16','B_3': 'float16','D_42': 'float16','D_43': 'float16','D_44': 'float16', 'B_4': 'float16','D_45': 'float16','B_5': 'float16','R_2': 'float16','D_46': 'float16','D_47': 'float16','D_48': 'float16', 'D_49': 'float16','B_6': 'float16','B_7': 'float16','B_8': 'float16','D_50': 'float16','D_51': 'float16','B_9': 'float16', 'R_3': 'float16','D_52': 'float16','P_3': 'float16','B_10': 'float16','D_53': 'float16','S_5': 'float16','B_11': 'float16', 'S_6': 'float16','D_54': 'float16','R_4': 'float16','S_7': 'float16','B_12': 'float16','S_8': 'float16','D_55': 'float16', 'D_56': 'float16','B_13': 'float16','R_5': 'float16','D_58': 'float16','S_9': 'float16','B_14': 'float16','D_59': 'float16', 'D_60': 'float16','D_61': 'float16','B_15': 'float16','S_11': 'float16','D_62': 'float16','D_63': 'object','D_64': 'object', 'D_65': 'float16','B_16': 'float16','B_17': 'float16','B_18': 'float16','B_19': 'float16','D_66': 'float16','B_20': 'float16', 'D_68': 'float16','S_12': 'float16','R_6': 'float16','S_13': 'float16','B_21': 'float16','D_69': 'float16','B_22': 'float16', 'D_70': 'float16','D_71': 'float16','D_72': 'float16','S_15': 'float16','B_23': 'float16','D_73': 'float16','P_4': 'float16', 'D_74': 'float16','D_75': 'float16','D_76': 'float16','B_24': 'float16','R_7': 'float16','D_77': 'float16','B_25': 'float16', 'B_26': 'float16','D_78': 'float16','D_79': 'float16','R_8': 'float16','R_9': 'float16','S_16': 'float16','D_80': 'float16', 'R_10': 'float16','R_11': 'float16','B_27': 'float16','D_81': 'float16','D_82': 'float16','S_17': 'float16','R_12': 'float16', 'B_28': 'float16','R_13': 'float16','D_83': 'float16','R_14': 'float16','R_15': 'float16','D_84': 'float16','R_16': 'float16', 'B_29': 'float16','B_30': 'float16','S_18': 'float16','D_86': 'float16','D_87': 'float16','R_17': 'float16','R_18': 'float16', 'D_88': 'float16','B_31': 'int64','S_19': 'float16','R_19': 'float16','B_32': 'float16','S_20': 'float16','R_20': 'float16', 'R_21': 'float16','B_33': 'float16','D_89': 'float16','R_22': 'float16','R_23': 'float16','D_91': 'float16','D_92': 'float16', 'D_93': 'float16','D_94': 'float16','R_24': 'float16','R_25': 'float16','D_96': 'float16','S_22': 'float16','S_23': 'float16', 'S_24': 'float16','S_25': 'float16','S_26': 'float16','D_102': 'float16','D_103': 'float16','D_104': 'float16','D_105': 'float16', 'D_106': 'float16','D_107': 'float16','B_36': 'float16','B_37': 'float16', 'R_26': 'float16','R_27': 'float16','B_38': 'float16', 'D_108': 'float16','D_109': 'float16','D_110': 'float16','D_111': 'float16','B_39': 'float16','D_112': 'float16','B_40': 'float16', 'S_27': 'float16','D_113': 'float16','D_114': 'float16','D_115': 'float16','D_116': 'float16','D_117': 'float16','D_118': 'float16', 'D_119': 'float16','D_120': 'float16','D_121': 'float16','D_122': 'float16','D_123': 'float16','D_124': 'float16','D_125': 'float16', 'D_126': 'float16','D_127': 'float16','D_128': 'float16','D_129': 'float16','B_41': 'float16','B_42': 'float16','D_130': 'float16', 'D_131': 'float16','D_132': 'float16','D_133': 'float16','R_28': 'float16','D_134': 'float16','D_135': 'float16','D_136': 'float16', 'D_137': 'float16','D_138': 'float16','D_139': 'float16','D_140': 'float16','D_141': 'float16','D_142': 'float16','D_143': 'float16', 'D_144': 'float16','D_145': 'float16'}

## Reading the data
train = pd.read_csv(file_content_stream)
# test = pd.read_csv(file_content_stream2, dtype = dtype_dict)

## Sanity check
print('-- Data Read -- \n')

## -------------------------------------------

#########################
## Training Data-Frame ##
#########################

## Sanity check
print('-- Data-frame imputation starting -- \n')

## Defining the input variables and dropping categorical variables
mf_train = train.drop(columns = ['customer_ID', 'target'])

# Building the miceforest kernel
kernel_train = mf.ImputationKernel(mf_train, datasets = 5, save_all_iterations = True)

## Assigning the final imputed data-frames
train_impute = kernel_train.complete_data(dataset = 0, inplace = False)

## Adding "customer_ID" back into the data-frames
train_impute = pd.concat([train[['customer_ID', 'target']], train_impute], axis = 1)

## Sanity check
print('-- Data-frame imputation complete -- \n')

## -------------------------------------------

## Subsetting the data to only include the most important features
train = train_impute[['customer_ID', 'B_1_median', 'B_9_sum', 'B_9_mean', 'B_2_mean', 'B_2_sum', 'B_3_data_range', 'B_9_median', 'B_18_sum', 'B_4_correlation', 'B_26_mean']]

## Sanity check
print('-- Data subsetting complete -- \n')

## -------------------------------------------

## Exporting the resulting training data-frame to the S3 Bucket
train.to_csv('amex_train_data_balance_final.csv', index = False)

sess.upload_data(path = 'amex_train_data_balance_final.csv', bucket = bucket_name, key_prefix = 'Kaggle-American-Express-Default')

## Sanity check
print('-- Train Complete -- \n')

## -------------------------------------------

# ########################
# ## Testing Data-Frame ##
# ########################

# ## Sanity check
# print('-- Test data-frame imputation starting -- \n')

# ## Defining the input variables and dropping categorical variables
# mf_test = test.drop(columns = ['customer_ID'])

# # Building the miceforest kernel
# kernel_test = mf.ImputationKernel(mf_test, datasets = 5, save_all_iterations = True)

# ## Assigning the final imputed data-frames
# test_impute = kernel_test.complete_data(dataset = 0, inplace = False)

# ## Adding "customer_ID" back into the data-frames
# test_impute = pd.concat([test[['customer_ID']], test_impute], axis = 1)

# ## Sanity check
# print('-- Test data-frame imputation complete -- \n')

# ## -------------------------------------------

# ## Sanity check
# print('-- Starting feature aggregations on the test data-frame -- \n')

# ## Creating aggregation functions
# def data_range(x):
#     return x.max() - x.min()

# def correlation(x):
#     return pd.Series(x.values).corr(other = pd.Series(x.index), method = 'pearson')

# ## Creating new Payment features with the cleaned test data-frame
# aggregated_vars_test = test_impute.groupby('customer_ID').agg({'B_1':['median'], 'B_2':['mean', 'sum'], 'B_3':[data_range], 'B_4':[correlation], 'B_9':['mean', 'median', 'sum'], 'B_18':['sum'], 'B_26':['mean']}).reset_index(drop = False)

# ## Renaming variables
# aggregated_vars_test.columns = ['customer_ID', 'B_1_median', 'B_2_mean', 'B_2_sum', 'B_3_data_range', 'B_4_correlation', 'B_9_mean', 'B_9_median', 'B_9_sum', 'B_18_sum', 'B_26_mean']

# ## Sanity check
# print('-- Testing aggregations data-frame complete -- \n')

# ## -------------------------------------------

# ## Sanity check
# print('-- Final test data-frame imputation starting -- \n')

# ## Defining the input variables and dropping categorical variables
# mf_data = aggregated_vars_test.drop(columns = ['customer_ID'])

# # Building the miceforest kernel
# kernel_data = mf.ImputationKernel(mf_data, datasets = 5, save_all_iterations = True)

# ## Assigning the final imputed data-frames
# data_impute = kernel_data.complete_data(dataset = 0, inplace = False)

# ## Adding "customer_ID" back into the data-frames
# data_impute = pd.concat([aggregated_vars_test[['customer_ID']], data_impute], axis = 1)

# ## Sanity check
# print('-- Final test data-frame imputation complete -- \n')

# ## -------------------------------------------

# ## Exporting the resulting training data-frame to the S3 Bucket
# data_impute.to_csv('amex_test_data_balance_final.csv', index = False)

# sess.upload_data(path = 'amex_test_data_balance_final.csv', bucket = bucket_name, key_prefix = 'Kaggle-American-Express-Default')

# ## Sanity check
# print('-- Test Complete -- \n')