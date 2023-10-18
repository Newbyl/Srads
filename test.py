import numpy as np
from autodiff import *
from srads import *


a = Variable(np.array([2, 2]))
b = Variable(np.array([2, 2]))
c = Variable(np.array([2]))
y = a * b + c

# Compute the gradients of the output variable with respect to the inputs
gradients = get_gradients(y)

# Print the gradients
print(y.array)
print('grad_a:', gradients[a])
print('grad_b:', gradients[b])
print('grad_c:', gradients[c])

L1 = Dense(2, 2, relu)
L1.forward(Variable(np.array([[1, 2], 
                    [3, 4]])))

gradients = get_gradients(L1.output)

print('grads_weights', gradients[L1.weights])
print('grads_bias',gradients[L1.bias])

model = Model()
model.add(Dense(2, 2, relu))
model.add(Dense(2, 2, relu))
model.add(Dense(2, 1, relu))

model.compile(cross_entropy, cross_entropy)

print(model.fit(np.array([[1, 2]]), np.array([[1], [0]]), 1))
