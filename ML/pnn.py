from keras.models import Sequential
from keras.layers import Dense, Lambda
import keras.backend as K
import tensorflow as tf

class ProbabilisticNeuralNetwork:
    def __init__(self, input_dim, num_classes, num_components=5):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_components = num_components
        self.model = self.probabilistic_nn()

    def probabilistic_nn(self):
        #build a pnn with 5 layers
        model = Sequential()
        model.add(Dense(41, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(20, activation='relu'))
        #create an layer for managing Gaussian distribution components (mu, log_sigma, pi)
        model.add(Dense(self.num_components * 3, activation=None))
        #probabilities normalization
        model.add(Lambda(lambda x: x / K.sum(x, axis=-1, keepdims=True)))
        model.add(Dense(self.num_classes, activation='softmax'))  # Output layer con softmax per la classificazione multiclasse
        model.compile(loss=self.loss_fun, optimizer='adam', metrics=['accuracy'])
        return model

    def loss_fun(self, y_true, y_pred):
        #division of the prediction into three components
        components = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(y_pred)
        # mu = expected average value
        #log_sigma=log of the standard deviation of the distribution(how much the data tends to deviate from the average)
        mu, log_sigma, pi = components

        y_true = tf.cast(y_true, dtype=tf.float32)

        #calculate the exponent in the Gaussian distribution formula evaluating 
        #of how close the mu values are to the actual value
        exponent = -0.5 * K.sum(K.square((y_true - mu) / (K.exp(log_sigma) + K.epsilon())), axis=-1)
        #normalize the pi(weight for each component) coefficients
        normalizer = K.sum(pi, axis=-1, keepdims=True)
        log_probs = K.log(K.maximum(normalizer, K.epsilon())) + exponent + K.log(pi)

        loss = -K.mean(log_probs)
        return loss

    def train(self, x_train, y_train, epochs=50, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate(self, x_test, y_test, verbose):
        accuracy = self.model.evaluate(x_test, y_test, verbose=verbose)[1]
        return accuracy
        