import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Generate a logistic regression dataset
X, y = make_classification(
    n_classes=2,
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)
y = y[:, np.newaxis]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class LogisticRegression:
    def __init__(self, learning_rate=0.1, ):
        self.lr = learning_rate

    def train(self, X, y, epochs=100):
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0
        for it in range(epochs):
            y_predict = self.predict(X)
            dw = np.dot(X.T, (y_predict - y)) / X.shape[0]
            db = np.sum(y_predict - y) / X.shape[0]

            self.weights -= dw
            self.bias -= db

        return self.weights, self.bias

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


model = LogisticRegression()
model.train(X_train, y_train)
y_pred = (model.predict(X_test)[:,0]>0.5).astype(int)
accuracy = metrics.accuracy_score(y_pred, y_test[:,0])
print('Accuracy:',accuracy)