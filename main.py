import os
import numpy as numpy
import pandas as pd
from dataset_features import features
from attacks_categories import u2r_attacks, r2l_attacks

#get current work directory
cwd = os.getcwd()

#set dataset train and test path
path_DS_Train = cwd + "\\NSL-KDD\\KDDTrain+.txt"
path_DS_Test = cwd + "\\NSL-KDD\\KDDTest+.txt"

#read train and test data
df_train = pd.read_csv(path_DS_Train)
df_test = pd.read_csv(path_DS_Test)
#print("df_train rows : " + str(len(df_train)), "- df_train columns : " + str(len(df_train.columns)))
#print("df_test rows : " + str(len(df_test)), "- df_test columns : " + str(len(df_test.columns)))

df_train.columns = features

#deleted u2r and r2l attacks rows
df_train = df_train[~df_train['outcome'].isin(u2r_attacks) & ~df_train['outcome'].isin(r2l_attacks)]
print("df_train rows : " + str(len(df_train)), "- df_train columns : " + str(len(df_train.columns)))
#df_train_useless_value = df_train[df_train['outcome'].isin(u2r_attacks) | df_train['outcome'].isin(r2l_attacks)]
#print("df_train rows : " + str(len(df_train_useless_value)), "- df_train columns : " + str(len(df_train_useless_value.columns)))