import os

def get_dataframe_split(df):
    # get all the features
    x = df.iloc[:, :-1]
    #get label
    y = df.iloc[:, -1]

    return x,y

def get_dataset_path(type):

    cwd = os.getcwd()
    pathDS = cwd + "\\NSL-KDD\\KDD" + type

    return pathDS