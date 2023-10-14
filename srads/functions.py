import numpy as np

def relu(Z):
    A = np.maximum(0,Z)
    
    return A

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    
    return A

def softmax(Z):
    exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def binary_cross_entropy(Y, a_out):
    m = Y.shape[1]
    return  (-1/m) * (np.dot(Y, np.log(a_out).T) + np.dot((1-Y), np.log(1-a_out).T))