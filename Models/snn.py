from keras.models import Sequential
from keras.layers import Dense
import shap

class SequentialNeuralNetwork(Sequential):
    def __init__(self, input_dim, num_classes, activation, loss_fun):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = activation
        self.loss_fun = loss_fun
        self.build_model()

    def build_model(self):
        self.add(Dense(41, input_dim=self.input_dim, activation='relu'))
        self.add(Dense(20, activation='relu'))
        self.add(Dense(units=self.num_classes, activation=self.activation)) 
        self.compile(loss=self.loss_fun, optimizer='adam', metrics=['accuracy'])
    
    def plot_important_features(self, x_test, class_index=None):
        explainer = shap.KernelExplainer(self.predict, x_test)
        shap_values = explainer.shap_values(x_test)
        if class_index is not None:
            shap.summary_plot(shap_values[class_index], x_test, class_names=["Normale", "DoS", "Probe"], max_display=x_test.shape[1])
        else:
            shap.summary_plot(shap_values, x_test, class_names=["Normale", "DoS", "Probe"], max_display=x_test.shape[1])

