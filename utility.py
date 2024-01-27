import os
import pandas as pd
from dataset_features import features

def get_dataframe_split(dataset_path, header=False):
    # load df from csv
    df = pd.read_csv(dataset_path)
    #set column name if requested
    if(header):
        df.columns = features[:-1]
    # get all the features
    x = df.iloc[:, :-1]
    #get label
    y = df.iloc[:, -1]

    return x,y

def get_dataset_path(type):
    # get current working directory
    cwd = os.getcwd()
    # get path to dataset directory
    pathDS = cwd + "\\NSL-KDD\\KDD" + type

    return pathDS