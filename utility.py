import os
import pandas as pd

def get_dataframe_split(dataset_path):
    # load df from csv
    df = pd.read_csv(dataset_path)
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