import numpy as np
from autodiff import *
from typing import Callable


class functions():
    ...


class Initialiser():
    def __init__(self) -> None:
        pass

    def he_initialiser(self, input_shape : int, nb_n : int) -> np.ndarray:
        return np.random.randn(input_shape, nb_n) * np.sqrt(2 / input_shape)
    
    def normal_initialiser(self, input_shape : int, nb_n : int) -> np.ndarray:
        return np.random.randn(input_shape, nb_n)
    
    def xavier_initialiser(self, input_shape : int, nb_n : int) -> np.ndarray:
        return np.random.randn(input_shape, nb_n) * np.sqrt(1/input_shape)

    
class Layer:
    def __init__(self, input_shape : int, nb_n : int, activation : str) -> None:
        self.nb_n = nb_n
        self.activation = activation
        self.input_shape = input_shape

            
    
class Dense(Layer):
    def __init__(self, input_shape: int, nb_n: int, activation: Callable[[Variable], Variable], weights: np.ndarray = None, bias: np.ndarray = None) -> None:
        super().__init__(input_shape, nb_n, activation)
        self.weights = Variable(weights) if weights is not None else Variable(np.random.randn(input_shape, nb_n) * np.sqrt(2 / input_shape))
        self.bias = Variable(bias) if bias is not None else Variable(np.random.randn(1, nb_n) * 0.01)
        self.output = None
    
    def forward(self, inp : Variable) -> Variable:
        new_inp = inp
        self.output = matmul(new_inp, self.weights) + self.bias
        self.output = self.activation(self.output)
        return self.output
    


class Model():
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.optimizer = None
    
    
    def add(self, layer : Layer) -> None:
        self.layers.append(layer)
        
        
    def compile(self, loss : Callable[[Variable], Variable], optimizer : Callable) -> None:
        self.loss = loss
        self.optimizer = optimizer
        
        
    def fit(self, X : np.ndarray, Y : np.ndarray, epochs : int):
        X_var = Variable(X)
        forward = self.layers[0].forward(X_var)
        
        for layer in self.layers[1:]:
            forward = layer.forward(forward)
            
        loss = self.loss(forward, Y)
        gradients = get_gradients(loss)
        
        print(gradients)
        

        