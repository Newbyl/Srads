import numpy as np

class Initialiser():
    def __init__(self) -> None:
        pass

    def he_initialiser(self, input_size : int, nb_n : int) -> np.ndarray:
        return np.random.randn(nb_n, input_size) * np.sqrt(2 / input_size)
    
    def normal_initialiser(self, input_size : int, nb_n : int) -> np.ndarray:
        return np.random.randn(nb_n, input_size)
    
    def xavier_initialiser(self, input_size : int, nb_n : int) -> np.ndarray:
        return np.random.randn(nb_n, input_size) * np.sqrt(1/input_size)
