import numpy as np
from functions import Function
from initialisers import Initialiser

sequence = None

class Layer:
	def __init__(self):
		self.name = self.__class__.__name__
		self.param = 0
		self.activation = Function.identity
		self.input_layer = None
		self.output_layers = []

	def __str__(self):
		return self.name + super().__str__()


class Dense(Layer):
    def __init__(self, nb_n, input_shape=None, weights=None, biases=None, activation=Function.identity, name=None):
        super().__init__()
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        if input_shape is None:
            input_shape = sequence.get_inp_shape()

        self.activation = activation

        if weights is None:
            self.weights = Initialiser.he_initialiser(self.input_shape, nb_n)
        else:
            if weights.shape != (self.input_shape, nb_n):
                raise Exception("Wrong weights shape : "+ weights.shape + " instead of " + str((self.input_shape, nb_n)))
            else:
                self.weights = weights
        if biases is None:
            self.biases = np.zeros((1, nb_n))
        else:
            if biases.shape != (1, nb_n):
                raise Exception("Wrong biases shape : "+ biases.shape + " instead of " + str((1, nb_n)))
            else:
                self.biases = biases
        self.kernels = self.weights
        self.w_m = 0
        self.w_v = 0
        self.b_m = 0
        self.b_v = 0
        self.shape = (None, nb_n)
        self.param = self.input_shape * nb_n + nb_n
        self.not_softmax_cross_entrp = True
        if self.activation == Function.identity:
            self.notIdentity = False
        else:
            self.notIdentity = True

    def forward(self, inp, training=True):
        self.inp = inp
        self.z_out = np.dot(inp, self.weights) + self.biases
        self.a_out = self.activation(self.z_out)
        return self.a_out

    def backprop(self, grads, layer=1):
        if self.notIdentity and self.not_softmax_cross_entrp:
            grads *= self.activation(self.z_out, self.a_out, derivative=True)
        d_c_b = grads
        self.d_c_w = np.dot(self.inp.T, d_c_b)
        if layer:
            d_c_a = np.dot(d_c_b, self.weights.T)
        else:
            d_c_a = 0
        self.d_c_b = d_c_b.sum(axis=0, keepdims=True)
        return d_c_a

    class InputLayer(Layer): 
        def __init__(self, shape=None):
            super().__init__()
            self.name = self.__class__.__name__
            try:
                self.shape = (None, *shape)
            except:
                self.shape = (None, shape)
            self.param = 0
            self.activation = Function.identity