from NN.layer_config import LayerConfig
from NN.layer import Layer
from NN.perceptron import Perceptron

class NeuralNetwork:
    def __init__(self, layer_configs: list[LayerConfig]) -> None:
        self.__layers : list[Layer] = [Layer(layer_configs[i].get_size(), layer_configs[i].get_activation(), 1 if i == 0 else layer_configs[i - 1].get_size()) for i in range(len(layer_configs))]
    
    def predict(self, input: list[float]) -> list[float]:
        for i in range(len(self.__layers)): 
            if i == 0:
                self.__layers[i].input_pass(input)
            else:
                self.__layers[i].forward_pass(self.__layers[i - 1])
        return self.__layers[len(self.__layers) - 1].get_memory()
    
    def __str__(self) -> str:
        s = "Neural network : \n"
        for layer in self.__layers:
            s += str(layer)
        return s
        