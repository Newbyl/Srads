import numpy as np
from collections import defaultdict
from typing import Callable

"""

Minimal reverse-mode automatic differentiation to compute the gradients automatically

Check : https://sidsite.com/posts/autodiff/ to understand how it works

"""

class Variable:
    def __init__(
        self,
        array: np.ndarray,
        local_gradients: list[tuple['Variable', Callable]] | None = None,
    ) -> None:
        self.array = array
        self.local_gradients = local_gradients if local_gradients else []
        
        Variable.__add__ = add
        Variable.__mul__ = mul
        Variable.__neg__ = neg
        Variable.__sub__ = sub
        Variable.__truediv__ = div


def get_gradients(variable: Variable) -> dict[Variable, np.ndarray]:
    gradients = defaultdict(lambda: 0)

    def compute_gradients(variable, path_value):
        for child_variable, multiply_by_locgrad in variable.local_gradients:
            value_of_path_to_child = multiply_by_locgrad(path_value)
            gradients[child_variable] += value_of_path_to_child
            compute_gradients(child_variable, value_of_path_to_child)

    gradients[variable] = np.ones(variable.array.shape, variable.array.dtype)
    compute_gradients(variable, gradients[variable])
    return dict(gradients)


def enable_broadcast(
    a: Variable, b: Variable, matmul=False
) -> tuple[Variable, Variable]:
    "Enables gradients to be calculated when broadcasting."
    a_shape = a.array.shape[:-2] if matmul else a.array.shape
    b_shape = b.array.shape[:-2] if matmul else b.array.shape
    a_repeatdims, b_repeatdims = broadcastinfo(a_shape, b_shape)

    def multiply_by_locgrad_a(path_value):
        path_value = np.sum(path_value, axis=a_repeatdims).reshape(a.array.shape)
        return np.zeros(a.array.shape, a.array.dtype) + path_value

    def multiply_by_locgrad_b(path_value):
        path_value = np.sum(path_value, axis=b_repeatdims).reshape(b.array.shape)
        return np.zeros(b.array.shape, b.array.dtype) + path_value

    a_ = Variable(a.array, local_gradients=[(a, multiply_by_locgrad_a)])
    b_ = Variable(b.array, local_gradients=[(b, multiply_by_locgrad_b)])
    return a_, b_


def getitem(a: Variable, indices: np.ndarray) -> Variable:
    "Get elements of `a` using NumPy indexing."
    value = a.array[indices]

    def multiply_by_locgrad(path_value):
        "(Takes into account elements indexed multiple times.)"
        result = np.zeros(a.array.shape, a.array.dtype)
        np.add.at(result, indices, path_value)
        return result

    local_gradients = [(a, multiply_by_locgrad)]
    return Variable(value, local_gradients)


def broadcastinfo(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    "Get which dimensions are added or repeated when `a` and `b` are broadcast."
    ndim = max(len(a_shape), len(b_shape))

    add_ndims_to_a = ndim - len(a_shape)
    add_ndims_to_b = ndim - len(b_shape)

    a_shape_ = np.array([1] * add_ndims_to_a + list(a_shape))
    b_shape_ = np.array([1] * add_ndims_to_b + list(b_shape))

    if not all((a_shape_ == b_shape_) | (a_shape_ == 1) | (b_shape_ == 1)):
        raise ValueError(f"could not broadcast shapes {a_shape} {b_shape}")

    a_repeatdims = (a_shape_ == 1) & (b_shape_ > 1)  # the repeated dims
    a_repeatdims[:add_ndims_to_a] = True  # the added dims
    a_repeatdims = np.where(a_repeatdims == True)[0]  # indices of axes where True
    a_repeatdims = [int(i) for i in a_repeatdims]

    b_repeatdims = (b_shape_ == 1) & (a_shape_ > 1)
    b_repeatdims[:add_ndims_to_b] = True
    b_repeatdims = np.where(b_repeatdims == True)[0]
    b_repeatdims = [int(i) for i in b_repeatdims]

    return tuple(a_repeatdims), tuple(b_repeatdims)


def add(a: Variable, b: Variable) -> Variable:
    "Elementwise addition."
    value = a.array + b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = [
        (a_, lambda path_value: path_value),
        (b_, lambda path_value: path_value),
    ]
    return Variable(value, local_gradients)


def div(a: Variable, b: Variable) -> Variable:
    "Elementwise division."
    value = a.array / b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = [
        (a_, lambda path_value: path_value / b.array),
        (b_, lambda path_value: -path_value * a.array / np.square(b.array)),
    ]
    return Variable(value, local_gradients)


def exp(a: Variable) -> Variable:
    "Elementwise exp of `a`."
    value = np.exp(a.array)
    local_gradients = [(a, lambda path_value: path_value * np.exp(a.array))]
    return Variable(value, local_gradients)


def log(a: Variable) -> Variable:
    "Elementwise log of `a`."
    value = np.log(a.array)
    local_gradients = [(a, lambda path_value: path_value / a.array)]
    return Variable(value, local_gradients)


def reshape(a: Variable, shape: tuple[int, ...]) -> Variable:
    "Reshape `a` into shape `shape`."
    value = np.reshape(a.array, shape)
    local_gradients = [(a, lambda path_value: path_value.reshape(a.array.shape))]
    return Variable(value, local_gradients)


def matmul(a: Variable, b: Variable) -> Variable:
    "Matrix multiplication."
    value = np.matmul(a.array, b.array)
    a_, b_ = enable_broadcast(a, b, matmul=True)
    local_gradients = [
        (a_, lambda path_value: np.matmul(path_value, np.swapaxes(b.array, -2, -1))),
        (b_, lambda path_value: np.matmul(np.swapaxes(a.array, -2, -1), path_value)),
    ]
    return Variable(value, local_gradients)


def matrix_transpose(a: Variable) -> Variable:
    "Swap the end two axes."
    value = np.swapaxes(a.array, -2, -1)
    local_gradients = [(a, lambda path_value: np.swapaxes(path_value, -2, -1))]
    return Variable(value, local_gradients)


def mul(a: Variable, b: Variable) -> Variable:
    "Elementwise multiplication."
    value = a.array * b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = [
        (a_, lambda path_value: path_value * b.array),
        (b_, lambda path_value: path_value * a.array),
    ]
    return Variable(value, local_gradients)


def neg(a: Variable) -> Variable:
    "Negate a variable."
    value = -a.array
    local_gradients = [(a, lambda path_value: -path_value)]
    return Variable(value, local_gradients)


def square(a: Variable) -> Variable:
    "Square each element of `a`"
    value = np.square(a.array)

    def multiply_by_locgrad(path_value):
        return path_value * 2 * a.array

    local_gradients = [(a, multiply_by_locgrad)]
    return Variable(value, local_gradients)

def sub(a: Variable, b: Variable) -> Variable:
    "Elementwise subtraction."
    value = a.array - b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = [
        (a_, lambda path_value: path_value),
        (b_, lambda path_value: -path_value),
    ]
    return Variable(value, local_gradients)


def sum(a: Variable, axis: tuple[int, ...] | None = None) -> Variable:
    "Sum elements of `a`, along axes specified in `axis`."
    value = np.sum(a.array, axis)

    def multiply_by_locgrad(path_value):
        result = np.zeros(a.array.shape, a.array.dtype)
        if axis:  # Expand dims so they can be broadcast.
            path_value = np.expand_dims(path_value, axis)
        return result + path_value

    local_gradients = [(a, multiply_by_locgrad)]
    return Variable(value, local_gradients)


def softmax(a: Variable, axis: int = -1) -> Variable:
    "Softmax on `axis`."
    exp_a = exp(a - Variable(np.max(a.array)))
    sum_shape = list(a.shape)
    sum_shape[axis] = 1
    return exp_a / reshape(sum(exp_a, axis=axis), sum_shape)


def cross_entropy(y_pred: Variable, y_true: np.array, axis: int = -1) -> Variable:
    "Cross entropy loss."
    indices = (np.arange(len(y_true)), y_true)
    return neg(sum(log(getitem(y_pred, indices))))

