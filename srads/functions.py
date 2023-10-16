import numpy as np


class Function():
    def identity(self, z):
        return z
    
    def sigmoid(self, z, a=None, derivative=False):
        abs_signal = (1 + np.abs(z))
        if derivative:
            return 0.5 / abs_signal ** 2
        else:
            return 0.5 / abs_signal + 0.5


    def relu(self, z, a=None, derivative=False):
        if derivative:
            return z > 0
        else:
            z[z < 0] = 0
            return z

    def tanh(self, z, a=None, derivative=False):
        if derivative:
            return 1 - a ** 2
        else:
            return np.tanh(z)


    def softmax(self, z, a=None, derivative=False):
        if derivative:
            return 1
        else:
            exps = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)


    def cross_entropy_with_logits(self, logits, labels, epsilon=1e-12):
        return -np.sum(labels * np.log(logits + epsilon), axis=0, keepdims=True)


    def cross_entropy(self, logits, labels, epsilon=1e-12):
        labels = labels.clip(epsilon, 1 - epsilon)
        logits = logits.clip(epsilon, 1 - epsilon)
        return -labels * np.log(logits) - (1 - labels) * np.log(1 - logits)


    def del_cross_sigmoid(self, logits, labels):
        return (logits - labels)


    def del_cross_soft(self, logits, labels):
        return (logits - labels)


    def mean_squared_error(self, logits, labels):
        return ((logits - labels) ** 2) / 2


    def del_mean_squared_error(self, logits, labels):
        return (logits - labels)


