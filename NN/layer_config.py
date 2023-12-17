from NN.enums.activation_functions import ActivationFunctions

class LayerConfig:
    def __init__(self, n : int, activation : ActivationFunctions) -> None:
        self.__size : int = n
        self.__activation : ActivationFunctions = activation

    def get_activation(self) -> ActivationFunctions:
        return self.__activation
    
    def get_size(self) -> int:
        return self.__size