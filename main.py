import numpy as numpy
import pandas as pd
from utility import get_dataset_path, get_dataframe_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#set dataset train and test path
pathDSTrain = get_dataset_path("TrainOnlyDoSProbe")
pathDSTest = get_dataset_path("TestOnlyDoSProbe")

df_train = pd.read_csv(pathDSTrain)
df_test = pd.read_csv(pathDSTest)

x_train , y_train = get_dataframe_split(df_train)
x_test , y_test = get_dataframe_split(df_test)

svm_clf = svm.SVC(kernel="rbf")
svm_clf.fit(x_train, y_train)
prediction = svm_clf.predict(x_test)

accuracy = accuracy_score(y_test, prediction)
print(accuracy)



