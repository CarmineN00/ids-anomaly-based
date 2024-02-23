from utility import get_dataset_path, get_dataframe_split
from Models.snn import SequentialNeuralNetwork
from Models.rnn import RecurrentNeuralNetwork
from Models.drnn import DeepRecurrentNeuralNetwork


#test accuracy on dataset with all features
'''pathDSTrain = get_dataset_path("TrainOnlyDoSProbe")
pathDSTest = get_dataset_path("TestOnlyDoSProbe")'''
#test accuracy on dataset with the 15 most important features selected by random forest
'''pathDSTrain = get_dataset_path("TrainRFImportantFeatures")
pathDSTest = get_dataset_path("TestRFImportantFeatures")'''
#test binary accuracy on dataset with all features
'''pathDSTrain = get_dataset_path("TrainBinaryOnlyDoSProbe")
pathDSTest = get_dataset_path("TestBinaryOnlyDoSProbe")'''
#test binary accuracy on dataset with the 15 most important features selected by random forest
'''pathDSTrain = get_dataset_path("TrainBinaryImportantFeatures")
pathDSTest = get_dataset_path("TestBinaryImportantFeatures")'''
#test accuracy on dataset with important features selected by shap on snn model
'''pathDSTrain = get_dataset_path("TrainSHAPSNNImportantFeatures")
pathDSTest = get_dataset_path("TestSHAPSNNImportantFeatures")'''
#test accuracy on dataset with important features selected by shap on rnn model
'''pathDSTrain = get_dataset_path("TrainSHAPRNNImportantFeatures")
pathDSTest = get_dataset_path("TestSHAPRNNImportantFeatures")'''
#test accuracy on dataset with important features selected by shap on drnn model
'''pathDSTrain = get_dataset_path("TrainSHAPDRNNImportantFeatures")
pathDSTest = get_dataset_path("TestSHAPDRNNImportantFeatures")'''
#test accuracy on dataset with impactful features selected by shap on snn model
'''pathDSTrain = get_dataset_path("TrainSHAPSNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestSHAPSNNImpactfulFeatures")'''
#test accuracy on dataset with impactful features selected by shap on rnn model
pathDSTrain = get_dataset_path("TrainSHAPRNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestSHAPRNNImpactfulFeatures")
#test accuracy on dataset with impactful features selected by shap on drnn model
'''pathDSTrain = get_dataset_path("TrainSHAPDRNNImpactfulFeatures")
pathDSTest = get_dataset_path("TestSHAPDRNNImpactfulFeatures")'''
#get train and test set
x_train , y_train = get_dataframe_split(pathDSTrain)
x_test , y_test = get_dataframe_split(pathDSTest)

# reshape for compatibility with LSTM layer 
x_train_reshaped = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_reshaped = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

#for binary classification -> num_class:1, activation: sigmoid, loss_fun: binary_crossentropy
#for multiclass classification -> num_class: 3, activation: softmax, loss_fun: sparse_categorical_crossentropy

#test snn 
snn = SequentialNeuralNetwork(x_train.shape[1], num_classes=3, activation="softmax", loss_fun="sparse_categorical_crossentropy")
snn.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)
ssn_loss, ssn_accuracy = snn.evaluate(x_test, y_test, verbose=1)
print(f"Accuracy: {ssn_accuracy * 100:.2f}% - Loss: {ssn_loss:.2f} on test set with SNN")

#test rnn 
rnn = RecurrentNeuralNetwork(x_train.shape[1], num_classes=3, activation="softmax", loss_fun="sparse_categorical_crossentropy")
rnn.fit(x_train_reshaped, y_train, epochs=5, batch_size=32,verbose=1)
rnn_loss, trnn_accuracy = rnn.evaluate(x_test_reshaped, y_test, verbose=1)
print(f"Accuracy: {trnn_accuracy * 100:.2f}% - Loss: {rnn_loss:.2f} on test set with SNN")

#test drnn
drnn = DeepRecurrentNeuralNetwork(x_train.shape[1], num_classes=3, activation="softmax", loss_fun="sparse_categorical_crossentropy")
drnn.fit(x_train_reshaped, y_train, epochs=5, batch_size=32,verbose=1)
drnn_loss, drnn_accuracy = drnn.evaluate(x_test_reshaped, y_test, verbose=1)
print(f"Accuracy: {drnn_accuracy * 100:.2f}% - Loss: {drnn_loss:.2f} on test set with SNN")










