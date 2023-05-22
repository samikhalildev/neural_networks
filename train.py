import numpy as np
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
from layers import DenseLayer, CNNLayer, RNNLayer, LSTMLayer
from activations import SigmoidActivation, ReLUActivation, SoftmaxActivation

'''
    NOTE

    For binary classification e.g. [0], [1]:
        - use Sigmoid in the last layer
        - use mean squared error to calculate loss

    For multiple classes e.g. [1,0,0,0]:
        - use Softmax in the last year
        - use cross-entropy to calculate loss
        - use np.argmax in the predict function
'''

# Define the dataset
X = np.array([[1, 1], [0, 1], [1, 1], [1, 0]])
y = np.array([[0], [1], [0], [1]])

# Split the dataset into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42)

print('Training data length', len(train_X))
print('Validation data length', len(val_X))

input_size = train_X.shape[1]
output_size = train_y.shape[1]
hidden_size = 128

epochs = 15
learning_rate = 0.003

# Create the neural network
model = NeuralNetwork(learning_rate)

# Add layers to the network
model.add_layer(DenseLayer(input_size, hidden_size,
                activation=SigmoidActivation()))
model.add_layer(DenseLayer(hidden_size, output_size,
                activation=SigmoidActivation()))
# model.add_layer(CNNLayer(input_shape=(28, 28, 1), num_filters=16, filter_size=(3, 3), activation=ReLUActivation())
# model.add_layer(RNNLayer(input_size, hidden_size,
#                 activation=SigmoidActivation()))
# model.add_layer(LSTMLayer(hidden_size, hidden_size))

# Train the network
model.fit(train_X, train_y, epochs, val_X=val_X, val_y=val_y)

# Make predictions
predictions = model.predict(val_X)
print('val_X:', val_X)
print('val_y:', val_y)
print('Predictions:', predictions)
