import numpy as np
import functions as fun


class Layer:
    def __init__(self, nb_n : int, activation : str) -> None:
        self.nb_n = nb_n
        self.activation = activation


class Dense(Layer):
    def __init__(self, nb_n : int, activation : str) -> None:
        super.__init__()
        self.nb_n = nb_n
        self.activation = activation


class Forward():
    def __init__(self, array_L : np.ndarray, input_size : int) -> None:
        self.array_L = array_L
        self.input_size = input_size

    def weights_init(self) -> dict:
        param = {}

        param['W1'] = np.random.randn(self.array_L[0].nb_n, self.input_size) / np.sqrt(self.input_size)
        param['b1'] = np.zeros((self.array_L[0].nb_n, 1))

        for l in range(1, len(self.array_L)):
            param['W' + str(l+1)] = np.random.randn(self.array_L[l].nb_n, self.array_L[l-1].nb_n) / np.sqrt(self.array_L[l-1])
            param['b' + str(l+1)] = np.zeros((self.array_L[l].nb_n, 1))

            assert(param['W' + str(l)].shape == (self.array_L[l], self.array_L[l-1]))
            assert(param['b' + str(l)].shape == (self.array_L[l], 1))

        return param

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
        

        