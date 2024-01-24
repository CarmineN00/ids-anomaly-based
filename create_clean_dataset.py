import os
import numpy as numpy
import pandas as pd
from dataset_features import features
from attacks_categories import u2r_attacks, r2l_attacks, dos_attacks, probe_attacks

def create_dataset_only_dos_probe_attack(datasetPath, pathWriteNewDS, datasetType):

    #read data
    df = pd.read_csv(datasetPath)
    #set columns name
    df.columns = features
    #deleted u2r and r2l attacks rows
    df = df[~df['outcome'].isin(u2r_attacks) & ~df['outcome'].isin(r2l_attacks)]
    #set outcome in dos_attac or prob_attack based on attack categories
    df.loc[df['outcome'].isin(dos_attacks), 'outcome'] = "dos_attack"
    df.loc[df['outcome'].isin(probe_attacks), 'outcome'] = "probe_attack"
    #conversion of dataframe to txt
    df.to_csv(pathWriteNewDS + "KDD" + datasetType + "OnlyDoSProbe", index=False, header=False)


#get current work directory
cwd = os.getcwd()

#set dataset train and test path
pathDSTrain = cwd + "\\NSL-KDD\\KDDTrain+.txt"
pathDSTest = cwd + "\\NSL-KDD\\KDDTest+.txt"

#set dataset path where to write modified dataset
pathWriteNewDS = cwd + "\\NSL-KDD\\"

create_dataset_only_dos_probe_attack(pathDSTrain,pathWriteNewDS, "Train")
create_dataset_only_dos_probe_attack(pathDSTest,pathWriteNewDS, "Test")

