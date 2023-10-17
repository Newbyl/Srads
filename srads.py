import numpy as np


class functions():
    ...


class Initialiser():
    def __init__(self) -> None:
        pass

    def he_initialiser(self, input_size : int, nb_n : int) -> np.ndarray:
        return np.random.randn(nb_n, input_size) * np.sqrt(2 / input_size)
    
    def normal_initialiser(self, input_size : int, nb_n : int) -> np.ndarray:
        return np.random.randn(nb_n, input_size)
    
    def xavier_initialiser(self, input_size : int, nb_n : int) -> np.ndarray:
        return np.random.randn(nb_n, input_size) * np.sqrt(1/input_size)

    
class Layer:
    def __init__(self, input_shape : int, nb_n : int, activation : str) -> None:
        self.nb_n = nb_n
        self.activation = activation
        self.input_shape = input_shape



class Dense(Layer):
    def __init__(self, nb_n : int, activation : str, input_shape : int, weights : np.ndarray = None, bias : np.ndarray = None) -> None:
        super.__init__()
        self.nb_n = nb_n
        self.activation = activation
        self.input_shape = input_shape
        self.weights = weights
        self.bias = bias

        if (self.weights == None):
            self.weights = Initialiser.he_initialiser()
        if (self.bias == None):
            self.bias = np.random.randn(1, nb_n) * 0.01
            
    
    def forward(self, input : np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.dot(input, self.weights.T) + self.bias
        self.output = functions.activation(self.output, self.activation)
        return self.output
    
    




class Model():
    ...
        

        