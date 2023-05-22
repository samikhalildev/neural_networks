from layers import Layer
import numpy as np


class AutoencoderLayer(Layer):
    def __init__(self, input_size, hidden_size, activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.W_encoder = np.random.randn(hidden_size, input_size)
        self.b_encoder = np.zeros((hidden_size, 1))
        self.W_decoder = np.random.randn(input_size, hidden_size)
        self.b_decoder = np.zeros((input_size, 1))

    def forward(self, X):
        self.input = X

        # Encode
        self.encoded = self.activation(
            np.dot(self.W_encoder, X) + self.b_encoder)

        # Decode
        self.decoded = np.dot(self.W_decoder, self.encoded) + self.b_decoder

        return self.decoded

    def backward(self, error, learning_rate):
        # Compute gradients for decoder
        delta_decoder = error
        self.W_decoder_gradient = np.dot(delta_decoder, self.encoded.T)
        self.b_decoder_gradient = np.sum(delta_decoder, axis=1, keepdims=True)

        # Compute gradients for encoder
        delta_encoder = np.dot(self.W_decoder.T, delta_decoder) * \
            self.activation_derivative(self.encoded)
        self.W_encoder_gradient = np.dot(delta_encoder, self.input.T)
        self.b_encoder_gradient = np.sum(delta_encoder, axis=1, keepdims=True)

        error_next = np.dot(self.W_encoder.T, delta_encoder)

        # Update parameters
        self.W_decoder -= learning_rate * self.W_decoder_gradient
        self.b_decoder -= learning_rate * self.b_decoder_gradient
        self.W_encoder -= learning_rate * self.W_encoder_gradient
        self.b_encoder -= learning_rate * self.b_encoder_gradient

        return error_next

    def activation_derivative(self, X):
        return self.activation.derivative(X)
