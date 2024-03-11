import os
import pandas as pd
from dataset_features import features
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_dataframe_split(dataset_path, header=False):
    # load df from csv
    df = pd.read_csv(dataset_path)
    df = df.reset_index(drop=True)
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

def confusion_matrix_metrics(confusion_matrix, classification_type, name):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=['normal', 'anomaly'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(name)
    if(classification_type == "Binary"):
        tp=confusion_matrix[0][0]
        tn=confusion_matrix[1][1]
        fn=confusion_matrix[1][0]
        fp=confusion_matrix[0][1]
    elif(classification_type == "Multiclass"):
        tp = confusion_matrix[0][0]
        tn = confusion_matrix[1][1] + confusion_matrix[2][2]  
        fn = confusion_matrix[1][0] + confusion_matrix[2][0]  
        fp = confusion_matrix[0][1] + confusion_matrix[0][2]  
    SENSITIVITY=tp/(tp+fn)*100
    SPECIFICITY=tn/(fp+tn)*100
    PPV=tp/(tp+fp)*100
    NPV=tn/(fn+tn)*100 
    print("SENSITIVITY:", SENSITIVITY, "SPECIFICITY:", SPECIFICITY, "PPV:", PPV, "NPV:", NPV)
    plt.show()
  