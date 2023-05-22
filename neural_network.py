import numpy as np


class NeuralNetwork:
    def __init__(self, initial_learning_rate):
        self.layers = []
        self.initial_learning_rate = initial_learning_rate

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, error, learning_rate):
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def fit(self, X, y, epochs, val_X=None, val_y=None):
        learning_rate = self.initial_learning_rate

        for epoch in range(epochs):
            output = self.forward(X)
            error = output - y
            self.backward(error, learning_rate)

            if val_X is not None and val_y is not None:
                val_loss, val_accuracy = self.evaluate(val_X, val_y)
                print(
                    f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    def evaluate(self, X, y):
        output = self.forward(X)
        loss = np.mean((output - y) ** 2)
        accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy

    def predict(self, X):
        output = self.forward(X)
        predicted_labels = np.where(output >= 0.5, 1, 0)
        return output
