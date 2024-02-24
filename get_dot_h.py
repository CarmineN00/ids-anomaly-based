from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import load_model
from tinymlgen import port
from utility import get_dataset_path, get_dataframe_split

'''pathDSTrain = get_dataset_path("TrainSHAPSNNImportantFeatures")
pathDSTest = get_dataset_path("TestSHAPSNNImportantFeatures")

x_train , y_train = get_dataframe_split(pathDSTrain)
x_test , y_test = get_dataframe_split(pathDSTest)

#avoid the custom layer problem
model = Sequential()
model.add(Dense(41, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(units=3, activation="softmax")) 
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
model.save("TrainedModels\\SHAPSNNImportantFeatures.h5")'''

#convert a trained model in .h module
snn = load_model("TrainedModels\\SHAPSNNImportantFeatures.h5")
c_code = port(snn, variable_name="snn_model", pretty_print=True)
print(c_code)


