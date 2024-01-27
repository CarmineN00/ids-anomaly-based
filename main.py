from utility import get_dataset_path, get_dataframe_split
from Models.pnn import ProbabilisticNeuralNetwork
from Models.snn import SequentialNeuralNetwork
from Models.rnn import RecurrentNeuralNetwork
from Models.drnn import DeepRecurrentNeuralNetwork
from Models.svm import SVM

#test accuracy on dataset with all features
pathDSTrain = get_dataset_path("TrainOnlyDoSProbe")
pathDSTest = get_dataset_path("TestOnlyDoSProbe")
#test accuracy on dataset with the 15 most important features selected by random forest
pathDSTrain = get_dataset_path("ImportantFeatures")
pathDSTest = get_dataset_path("ImportantFeatures")

#get train and test set
x_train , y_train = get_dataframe_split(pathDSTrain)
x_test , y_test = get_dataframe_split(pathDSTest)

# reshape for compatibility with LSTM layer 
x_train_reshaped = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_reshaped = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

'''#test svm
svm = SVM(kernel='rbf',C=1,gamma='scale')
svm.train(x_train, y_train)
svm_accuracy = svm.evaluate(x_test,y_test)
print(f"Accuracy on test set with SVM: {svm_accuracy * 100:.2f}%")

#test pnn 
pnn = ProbabilisticNeuralNetwork(input_dim=x_train.shape[1],num_classes=3, num_components=5)
pnn.train(x_train, y_train, epochs=10, batch_size=32)
pnn_accuracy = pnn.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy on test set with PNN: {pnn_accuracy * 100:.2f}%")'''

#test snn 
snn = SequentialNeuralNetwork(x_train.shape[1],num_classes=3)
snn.train(x_train, y_train, epochs=5, batch_size=32)
ssn_accuracy = snn.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy on test set with SNN: {ssn_accuracy * 100:.2f}%")

#test rnn 
rnn = RecurrentNeuralNetwork(x_train.shape[1], num_classes=3)
rnn.train(x_train_reshaped, y_train, epochs=5, batch_size=32)
rnn_accuracy = rnn.evaluate(x_test_reshaped, y_test, verbose=0)
print(f"Accuracy on test set with RNN: {rnn_accuracy * 100:.2f}%")

#test drnn
drnn = DeepRecurrentNeuralNetwork(x_train.shape[1], num_classes=3)
drnn.train(x_train_reshaped, y_train, epochs=5, batch_size=32)
drnn_accuracy = drnn.evaluate(x_test_reshaped, y_test, verbose=0)
print(f"Accuracy on test set with DRNN: {drnn_accuracy * 100:.2f}%")










