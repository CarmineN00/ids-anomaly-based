from keras.models import Sequential
from keras.layers import LSTM, Dense

class RecurrentNeuralNetwork:
    def __init__(self, input_dim, num_classes, activation, loss_fun):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = activation
        self.loss_fun = loss_fun
        self.model = self.recurrent_nn()

    def recurrent_nn(self):
        #build a rnn using Long Short-Term Memory layer, first layer with number of neurons equal to features
        #last one with number equal to classes number
        model = Sequential()
        model.add(LSTM(41, input_shape=(self.input_dim, 1), dropout=0.2))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(units=self.num_classes, activation=self.activation)) 

        model.compile(loss=self.loss_fun, optimizer='adam', metrics=['accuracy'])
        return model
    
    def train(self,x_train_reshaped, y_train, epochs=5, batch_size=32):
        self.model.fit(x_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate(self, x_test_reshaped, y_test, verbose):
        accuracy = self.model.evaluate(x_test_reshaped, y_test, verbose=0)[1]
        return accuracy