from utility import get_dataset_path, get_dataframe_split, confusion_matrix_metrics
from sklearn.metrics import confusion_matrix
from Models.snn import SequentialNeuralNetwork
from Models.rnn import RecurrentNeuralNetwork
from Models.drnn import DeepRecurrentNeuralNetwork
import numpy as np

#MULTICLASS CLASSIFICATION
#test accuracy on dataset with all features
'''pathDSTrain = get_dataset_path("TrainOnlyDoSProbe")
pathDSTest = get_dataset_path("TestOnlyDoSProbe")'''
#test accuracy on dataset with the 15 most important features selected by random forest
'''pathDSTrain = get_dataset_path("TrainRFImportantFeatures")
pathDSTest = get_dataset_path("TestRFImportantFeatures")'''
#test accuracy on dataset with impactful features selected by shap on snn model
'''pathDSTrain = get_dataset_path("TrainSHAPSNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestSHAPSNNImpactfulFeatures")'''
#test accuracy on dataset with impactful features selected by shap on rnn model
'''pathDSTrain = get_dataset_path("TrainSHAPRNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestSHAPRNNImpactfulFeatures")'''
#test accuracy on dataset with impactful features selected by shap on drnn model
'''pathDSTrain = get_dataset_path("TrainSHAPDRNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestSHAPDRNNImpactfulFeatures")'''

#BINARY CLASSIFICATION
#test binary accuracy on dataset with all features
'''pathDSTrain = get_dataset_path("TrainBinaryOnlyDoSProbe")
pathDSTest = get_dataset_path("TestBinaryOnlyDoSProbe")'''
#test binary accuracy on dataset with the 15 most important features selected by random forest
'''pathDSTrain = get_dataset_path("TrainBinaryRFImportantFeatures")
pathDSTest = get_dataset_path("TestBinaryRFImportantFeatures")'''
#test accuracy on dataset with impactful features selected by shap on snn model
'''pathDSTrain = get_dataset_path("TrainBinarySHAPSNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestBinarySHAPSNNImpactfulFeatures")'''
#test accuracy on dataset with impactful features selected by shap on rnn model
'''pathDSTrain = get_dataset_path("TrainBinarySHAPRNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestBinarySHAPRNNImpactfulFeatures")'''
#test accuracy on dataset with impactful features selected by shap on drnn model
pathDSTrain = get_dataset_path("TrainBinarySHAPDRNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestBinarySHAPDRNNImpactfulFeatures")


#get train and test set
x_train , y_train = get_dataframe_split(pathDSTrain)
x_test , y_test = get_dataframe_split(pathDSTest)

# reshape for compatibility with LSTM layer 
x_train_reshaped = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_reshaped = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

#for binary classification -> num_class:1, activation: sigmoid, loss_fun: binary_crossentropy
#for multiclass classification -> num_class: 3, activation: softmax, loss_fun: sparse_categorical_crossentropy
#for binary y_pred = (y_pred > 0.5).astype(int) 
#for multiclass y_pred = np.argmax(y_pred, axis=1), y_pred = np.where(y_pred == 0, 0, np.where(y_pred == 1, 1, 2))

#test snn 
'''snn = SequentialNeuralNetwork(x_train.shape[1], num_classes=1, activation="sigmoid", loss_fun="binary_crossentropy")
snn.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)
ssn_loss, ssn_accuracy = snn.evaluate(x_test, y_test, verbose=0)
print(f"Binary Classification SHAPSNNIF - Accuracy: {ssn_accuracy * 100:.2f}% - Loss: {ssn_loss:.2f} on test set with SNN")
y_pred = snn.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix_metrics(confusion_matrix=conf_matrix, classification_type="Binary", name="SNN Binary Classification SNNSHAPIF")'''

#test rnn 
'''rnn = RecurrentNeuralNetwork(x_train.shape[1], num_classes=1, activation="sigmoid", loss_fun="binary_crossentropy")
rnn.fit(x_train_reshaped, y_train, epochs=5, batch_size=32,verbose=1)
rnn_loss, trnn_accuracy = rnn.evaluate(x_test_reshaped, y_test, verbose=0)
print(f"Binary Classification SHAPRNNIF - Accuracy: {trnn_accuracy * 100:.2f}% - Loss: {rnn_loss:.2f} on test set with RNN")
y_pred = rnn.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix_metrics(confusion_matrix=conf_matrix, classification_type="Binary", name="RNN Binary Classification RNNSHAPIF")'''


#test drnn
'''drnn = DeepRecurrentNeuralNetwork(x_train.shape[1], num_classes=1, activation="sigmoid", loss_fun="binary_crossentropy")
drnn.fit(x_train_reshaped, y_train, epochs=5, batch_size=32,verbose=1)
drnn_loss, drnn_accuracy = drnn.evaluate(x_test_reshaped, y_test, verbose=0)
print(f"Binary Classification SHAPDRNNIF - Accuracy: {drnn_accuracy * 100:.2f}% - Loss: {drnn_loss:.2f} on test set with DRNN")
y_pred = drnn.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix_metrics(confusion_matrix=conf_matrix, classification_type="Binary", name="DRNN Binary Classification DRNNSHAPIF")'''










