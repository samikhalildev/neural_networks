import numpy as np


class Layer:
    def forward(self, X):
        pass

    def backward(self, error, learning_rate):
        pass


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        self.activation = activation

    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.activation(self.output)

    def backward(self, error, learning_rate):
        delta = error * self.activation_derivative(self.output)
        self.weights_gradient = np.dot(self.input.T, delta)
        self.biases_gradient = np.sum(delta, axis=0)
        error = np.dot(delta, self.weights.T)
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient
        return error

    def activation_derivative(self, X):
        return self.activation.derivative(X)


class CNNLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size)
        self.biases = np.zeros(output_channels)

    def forward(self, X):
        self.input = X
        batch_size, input_height, input_width, _ = X.shape
        self.batch_size = batch_size

        output_height = int(
            (input_height - self.kernel_size + 2 * self.padding) / self.stride) + 1
        output_width = int((input_width - self.kernel_size +
                           2 * self.padding) / self.stride) + 1
        self.output = np.zeros(
            (batch_size, output_height, output_width, self.output_channels))

        if self.padding > 0:
            X = np.pad(X, ((0, 0), (self.padding, self.padding),
                       (self.padding, self.padding), (0, 0)), mode='constant')

        for i in range(output_height):
            for j in range(output_width):
                receptive_field = X[:, i*self.stride:i*self.stride +
                                    self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :]
                self.output[:, i, j, :] = np.tensordot(
                    receptive_field, self.weights, axes=([3], [1])) + self.biases

        return self.output

    def backward(self, error, learning_rate):
        delta = error
        batch_size, input_height, input_width, _ = self.input.shape

        input_gradient = np.zeros_like(self.input)
        weights_gradient = np.zeros_like(self.weights)
        biases_gradient = np.zeros_like(self.biases)

        for i in range(self.output.shape[1]):
            for j in range(self.output.shape[2]):
                receptive_field = self.input[:, i*self.stride:i*self.stride +
                                             self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :]

                input_gradient[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j *
                               self.stride+self.kernel_size, :] += np.tensordot(delta, self.weights, axes=([3], [0]))
                weights_gradient += np.tensordot(receptive_field,
                                                 delta, axes=([0, 1, 2], [0, 1, 2]))
                biases_gradient += np.sum(delta, axis=(0, 1, 2))

        if self.padding > 0:
            input_gradient = input_gradient[:, self.padding:-
                                            self.padding, self.padding:-self.padding, :]

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient


class RNNLayer(Layer):
    def __init__(self, input_size, hidden_size, activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.Wxh = np.random.randn(hidden_size, input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))

        self.h_prev = None

    def forward(self, X):
        self.input = X

        # Compute hidden state
        self.h = self.activation(
            np.dot(X, self.Wxh.T) + np.dot(self.Whh, self.h_prev) + self.bh)

        return self.h

    def backward(self, error, learning_rate):
        # Compute gradients
        delta = error * self.activation_derivative(self.h)
        self.Wxh_gradient = np.dot(delta, self.input.T)
        self.Whh_gradient = np.dot(delta, self.h_prev.T)
        self.bh_gradient = np.sum(delta, axis=1, keepdims=True)

        error_next = np.dot(self.Whh.T, delta)

        # Update parameters
        self.Wxh -= learning_rate * self.Wxh_gradient
        self.Whh -= learning_rate * self.Whh_gradient
        self.bh -= learning_rate * self.bh_gradient

        return error_next

    def activation_derivative(self, X):
        return self.activation.derivative(X)


class LSTMLayer(Layer):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        self.c_prev = None
        self.h_prev = None

    def forward(self, X):
        self.input = X

        concat = np.concatenate((X, self.h_prev), axis=1)

        ft = self.activation(np.dot(self.Wf, concat) + self.bf)
        it = self.activation(np.dot(self.Wi, concat) + self.bi)
        ot = self.activation(np.dot(self.Wo, concat) + self.bo)
        ct = self.activation(np.dot(self.Wc, concat) + self.bc)

        self.c = ft * self.c_prev + it * ct
        self.h = ot * self.activation(self.c)

        return self.h

    def backward(self, error, learning_rate):
        delta_h = error * self.activation_derivative(self.h)
        delta_c = delta_h * \
            self.activation_derivative(self.c) * self.activation(self.c)

        self.Wf_gradient = np.dot(delta_c * self.c_prev, self.input.T)
        self.Wi_gradient = np.dot(delta_c * self.c_prev, self.input.T)
        self.Wo_gradient = np.dot(delta_h, self.input.T)
        self.Wc_gradient = np.dot(delta_c, self.input.T)

        self.bf_gradient = np.sum(delta_c * self.c_prev, axis=1, keepdims=True)
        self.bi_gradient = np.sum(delta_c * self.c_prev, axis=1, keepdims=True)
        self.bo_gradient = np.sum(delta_h, axis=1, keepdims=True)
        self.bc_gradient = np.sum(delta_c, axis=1, keepdims=True)

        error_next = (np.dot(self.Wf.T, delta_c * self.c_prev)
                      + np.dot(self.Wi.T, delta_c * self.c_prev)
                      + np.dot(self.Wo.T, delta_h)
                      + np.dot(self.Wc.T, delta_c))

        self.Wf -= learning_rate * self.Wf_gradient
        self.Wi -= learning_rate * self.Wi_gradient
        self.Wo -= learning_rate * self.Wo_gradient
        self.Wc -= learning_rate * self.Wc_gradient

        self.bf -= learning_rate * self.bf_gradient
        self.bi -= learning_rate * self.bi_gradient
        self.bo -= learning_rate * self.bo_gradient
        self.bc -= learning_rate * self.bc_gradient

        return error_next
