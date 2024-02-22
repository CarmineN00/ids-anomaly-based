import shap
from utility import get_dataset_path, get_dataframe_split
from Models.snn import SequentialNeuralNetwork
from Models.rnn import RecurrentNeuralNetwork
from Models.drnn import DeepRecurrentNeuralNetwork


pathDSTrain = get_dataset_path("TrainOnlyDoSProbe")
pathDSTest = get_dataset_path("TestOnlyDoSProbe")

x_train , y_train = get_dataframe_split(pathDSTrain)
x_test , y_test = get_dataframe_split(pathDSTest)

x_train_reshaped = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_reshaped = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

#shap value on first 50 rows of test set
x_test = x_test.head(5)

'''snn = SequentialNeuralNetwork(x_train.shape[1], num_classes=3, activation="softmax", loss_fun="sparse_categorical_crossentropy")
snn.fit(x_train, y_train, epochs=5, batch_size=32)
snn.plot_important_features(x_test)'''

'''rnn = RecurrentNeuralNetwork(x_train.shape[1], num_classes=3, activation="softmax", loss_fun="sparse_categorical_crossentropy")
rnn.fit(x_train_reshaped, y_train, epochs=5, batch_size=32)
rnn.plot_important_features(x_test)'''

'''drnn = DeepRecurrentNeuralNetwork(x_train.shape[1], num_classes=3, activation="softmax", loss_fun="sparse_categorical_crossentropy")
drnn.fit(x_train_reshaped, y_train, epochs=1, batch_size=32)
drnn.plot_important_features(x_test)'''

