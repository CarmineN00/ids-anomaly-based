from keras.models import Sequential
from keras.layers import Dense
import shap

class SequentialNeuralNetwork:
    def __init__(self, input_dim, num_classes, activation, loss_fun):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = activation
        self.loss_fun = loss_fun
        self.model = self.sequential_nn()

    def sequential_nn(self):
        #build a nn with 3 layer, first layer with number of neurons equal to features
        #last one with number equal to classes number
        model = Sequential()
        model.add(Dense(41, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(units=self.num_classes, activation=self.activation)) 
        model.compile(loss=self.loss_fun, optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=5, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate(self, x_test, y_test, verbose):
        accuracy = self.model.evaluate(x_test, y_test, verbose=verbose)[1]
        return accuracy
    
    def plot_important_features(self, x_test, class_index):
        explainer = shap.KernelExplainer(self.model.predict,x_test)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values[class_index], x_test, class_names=["Normale", "DoS", "Probe"], max_display=x_test.shape[1])
