
import math
import numpy as np
import numpy as np

class ANN:
    def __init__(self):
        self.weightdict = {}
        self.biasdict = {}
        self.activations = []
        self.neurons = []
        self.activationvalues = []
        self.layercount = 0
        self.initmethod = 'he'
        self.loss = 'mse'
        self.optimizer = 'adam'
        self.opt = Optimizer()

    def initializeweight(self, shape):
        fan_in = shape[0]
        fan_out = shape[1]

        if self.initmethod == 'he':
            limit = np.sqrt(2 / fan_in)
        elif self.initmethod == 'xavier':
            limit = np.sqrt(6 / (fan_in + fan_out))
        else:
            raise ValueError("Unsupported initialization method. Choose 'xavier' or 'he'.")

        return np.random.uniform(-limit, limit, size=shape)

    def add(self, neurons, activation, inputshape=None):
        if self.layercount == 0 and inputshape is not None:
            self.neurons.append(inputshape[1])

        self.neurons.append(neurons)
        self.activations.append(activation)

        self.weightdict[self.layercount] = self.initializeweight((self.neurons[-2], self.neurons[-1]))
        self.biasdict[self.layercount] = np.zeros((1, self.neurons[-1]))
        self.layercount += 1

    def activation(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'leakyrelu':
            return np.maximum(0.01 * x, x)
        elif activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported activation function.")

    def activationderivative(self, x, activation):
        if activation == 'linear':
            return np.ones_like(x)
        elif activation == 'sigmoid':
            sig = self.activation(x, 'sigmoid')
            return sig * (1 - sig)
        elif activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif activation == 'leakyrelu':
            return np.where(x > 0, 1, 0.01)
        else:
            raise ValueError("Unsupported activation function.")

    def forward(self, x):
        self.activationvalues = [x]
        for i in range(self.layercount):
            x = np.dot(x, self.weightdict[i]) + self.biasdict[i]
            x = self.activation(x, self.activations[i])
            self.activationvalues.append(x)
        return x

    def lossderivative(self, y_true, y_pred):
        if self.loss == 'mse':
            return 2 * (y_pred - y_true)
        elif self.loss in ['binary_crossentropy', 'categorical_crossentropy']:
            return y_pred - y_true
        else:
            raise ValueError("Unsupported loss function.")

    def backward(self, x, y, z, learning_rate):
        delta = self.lossderivative(y, z)

        for j in range(self.layercount - 1, -1, -1):
            delta *= self.activationderivative(self.activationvalues[j + 1], self.activations[j])

            dw = np.dot(self.activationvalues[j].T, delta) / x.shape[0]
            db = np.sum(delta, axis=0, keepdims=True) / x.shape[0]

            self.weightdict[j], self.biasdict[j] = self.opt.optimizer(
                self.optimizer, self.weightdict[j], dw, self.biasdict[j], db, learning_rate
            )

            if j > 0:
                delta = np.dot(delta, self.weightdict[j].T)

    def fit(self, x, y, epochs, learning_rate):

        for _ in range(epochs):
            z = self.forward(x)
            self.backward(x, y, z, learning_rate)
        return self.weightdict, self.biasdict

    def predict(self, x):

        return self.forward(x)


class Optimizer:

    def __init__(self):
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.t = 0

    def optimizer(self, optimizer, w, dw, b, db, learning_rate, beta1=0.9, beta2=0.99, eps=1e-8):
        self.t += 1

        if optimizer == 'adam':
            self.m_w = beta1 * self.m_w + (1 - beta1) * dw
            self.m_b = beta1 * self.m_b + (1 - beta1) * db
            self.v_w = beta2 * self.v_w + (1 - beta2) * dw ** 2
            self.v_b = beta2 * self.v_b + (1 - beta2) * db ** 2

            m_w_hat = self.m_w / (1 - beta1 ** self.t)
            m_b_hat = self.m_b / (1 - beta1 ** self.t)
            v_w_hat = self.v_w / (1 - beta2 ** self.t)
            v_b_hat = self.v_b / (1 - beta2 ** self.t)

            w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

        elif optimizer == 'rmsprop':
            self.v_w = beta1 * self.v_w + (1 - beta1) * dw ** 2
            self.v_b = beta1 * self.v_b + (1 - beta1) * db ** 2
            w -= learning_rate * dw / (np.sqrt(self.v_w) + eps)
            b -= learning_rate * db / (np.sqrt(self.v_b) + eps)

        elif optimizer == 'sgd':
            w -= learning_rate * dw
            b -= learning_rate * db

        elif optimizer == 'momentum':
            self.v_w = beta1 * self.v_w + learning_rate * dw
            self.v_b = beta1 * self.v_b + learning_rate * db
            w -= self.v_w
            b -= self.v_b

        else:
            raise ValueError("Unsupported optimizer function.")

        return w, b
