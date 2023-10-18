import numpy as np
from autodiff import *


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