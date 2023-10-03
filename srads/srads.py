import numpy as np
import functions as fun


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
            



class Forward():
    def __init__(self, array_L : np.ndarray) -> None:
        self.array_L = array_L


    def linear_forward(self, Activations : np.ndarray, Weights : np.ndarray, bias) -> tuple:
        Z = Weights.dot(Activations) + bias

        cache = (Activations, Weights, bias)

        return Z, cache
    
    def linear_activation_forward_relu(self, A_prev : np.ndarray, Weights : np.ndarray, bias : np.ndarray) -> tuple:
        Z, linearCache = self.linear_forward(A_prev, Weights, bias)
        A, activationCache = fun.relu(Z)

        cache = (linearCache, activationCache)

        return A, cache

    def linear_activation_forward_sigmoid(self, A_prev : np.ndarray, Weights : np.ndarray, bias : np.ndarray) -> tuple:
        Z, linearCache = self.linear_forward(A_prev, Weights, bias)
        A, activationCache = fun.sigmoid(Z)

        cache = (linearCache, activationCache)

        return A, cache
    
    def forward_propagation(self, X, param):
        caches = []
        A = X
        # We devide by 2 here because there are weights and biases
        # in the same dict param
        L = len(self.array_L) // 2

        A_prev = A

        # First layer
        if self.array_L.activation[0] == "relu":
            A, cache = self.linear_activation_forward_relu(A_prev, param['W1'], param['b1'])
            caches.append(cache)

        if self.array_L.activation[0] == "sigmoid":
            A, cache = self.linear_activation_forward_sigmoid(A_prev, param['W1'], param['b1'])
            caches.append(cache)

        for l in range(1, L-1):
            A_prev = A
            
            if self.array_L.activation == "relu":
                A, cache = self.linear_activation_forward_relu(A_prev, param['W' + str(l+1)], param['b' + str(l+1)])
                caches.append(cache)

            if self.array_L.activation == "sigmoid":
                A, cache = self.linear_activation_forward_sigmoid(A_prev, param['W' + str(l+1)], param['b' + str(l+1)])
                caches.append(cache)

        # Last layer
        if self.array_L[L].activation == "relu":
            A_last, cache = self.linear_activation_forward_relu(A, param['W' + str(L)], param['b' + str(L)])
            caches.append(cache)

        if self.array_L[L].activation == "sigmoid":
            A_last, cache = self.linear_activation_forward_sigmoid(A_prev, param['W' + str(L)], param['b' + str(L)])
            caches.append(cache)

        assert(A_last.shape == (1,X.shape[0]))

        return A_last, caches


class Model():
    ...
        

        