from collections import defaultdict
import numpy as np

class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients
    
    def __add__(self, other):
        return add(self, other)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __sub__(self, other):
        return add(self, neg(other))

    def __truediv__(self, other):
        return mul(self, inv(other))
    
    def sin(self):
        value = np.sin(self.value)
        local_gradients = (
            (self, np.cos(self.value)),
        )
        return Variable(value, local_gradients)
    
    def exp(self):
        value = np.exp(self.value)
        local_gradients = (
            (self, value),
        )
        return Variable(value, local_gradients)
    
    def log(self):
        value = np.log(self.value)
        local_gradients = (
            (self, 1. / self.value),
        )
        return Variable(value, local_gradients)
    
    def get_gradients(variable):
        """ Compute the first derivatives of `variable` 
        with respect to child variables.
        """
        gradients = defaultdict(lambda: 0)
        
        def compute_gradients(variable, path_value):
            for child_variable, local_gradient in variable.local_gradients:
                # "Multiply the edges of a path":
                value_of_path_to_child = path_value * local_gradient
                # "Add together the different paths":
                gradients[child_variable] += value_of_path_to_child
                # recurse through graph:
                compute_gradients(child_variable, value_of_path_to_child)
        
        gradients[variable] = np.ones(variable.value.shape, variable.value.dtype)
        compute_gradients(variable, gradients[variable])
        return dict(gradients)

def add(a, b):
    value = a.value + b.value    
    local_gradients = (
        (a, 1),
        (b, 1)
    )
    return Variable(value, local_gradients)

def mul(a, b):
    value = a.value * b.value
    local_gradients = (
        (a, b.value),
        (b, a.value)
    )
    return Variable(value, local_gradients)

def neg(a):
    value = -1 * a.value
    local_gradients = (
        (a, -1),
    )
    return Variable(value, local_gradients)

def inv(a):
    value = 1. / a.value
    local_gradients = (
        (a, -1 * value**2),
    )
    return Variable(value, local_gradients)