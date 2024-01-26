from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class SVM:
    def __init__(self, kernel, C, gamma):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy