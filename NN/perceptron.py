from NN.exceptions.invalid_input_size_exception import InvalidInputSizeException
from NN.enums.activation_functions import ActivationFunctions
import random
import numpy as np

import json
from json import JSONEncoder
from numpy_array_encoder import NumpyArrayEncoder

class Perceptron: 
    def __init__(self, n: int, activation: ActivationFunctions, is_input = False) -> None:
        LOW = -0.5
        HIGH = 0.5
        if activation == ActivationFunctions.Logistic:
            LOW = 0
            HIGH = 1 / n * 10
        if not is_input: 
            self.__weights : np.ndarray[float] = [random.uniform(LOW, HIGH) for _ in range(n)]
            self.__bias : float = random.uniform(LOW, HIGH)
        else: 
            self.__weights : np.ndarray[float] = [0.5 for _ in range(n)]
            self.__bias : float = 0.0
        self.__activation : ActivationFunctions = activation

    def get_weights(self): 
        return self.__weights

    def diff(self, output : float, expected : float) -> float:
        return 2 * (expected - output)
    
    def activate(self, input : float) -> float:
        return self.__activation.value[0](input)
    
    def differentiate(self, input : float) -> float:
        return self.__activation.value[1](input)
    
    def predict(self, inputs : np.ndarray[float]) -> float: 
        if len(inputs) != len(self.__weights):
            raise InvalidInputSizeException()
        
        result : float = self.__bias
        for i in range(len(self.__weights)):
            result += self.__weights[i] * inputs[i]

            if self.__weights[i] != 1: 
                i = i

        return self.activate(result)

    def evaluate(self, inputs : np.ndarray[np.ndarray[float]], outputs : np.ndarray[float]) -> float:
        if len(outputs) == 0:
            return 0
        
        loss : float = 0
        for i in range(len(self.__weights)):
            o : float = self.predict(inputs[i])
            loss += self.diff(o, outputs[i])

        return abs(loss / len(outputs))
    
    def train(self, inputs : np.ndarray[np.ndarray[float]], outputs : np.ndarray[float], learning_rate : float = 0.001) -> None:
        if len(inputs) != len(outputs):
            raise InvalidInputSizeException()

        for i in range(len(inputs)):
            sum_of_inputs : float = 1
            for input in inputs[i]:
                sum_of_inputs += input
            
            prediction : float = self.predict(inputs[i])
            diff : float = abs(self.diff(prediction, outputs[i]))
            direction : float = 1.0
            if prediction > outputs[i]:
                direction = -1.0
            
            for j in range(len(self.__weights)):
                self.__weights[j] += direction * diff * (abs(inputs[i][j]) / sum_of_inputs) * learning_rate

            self.__bias += direction * diff * (1 / sum_of_inputs) * learning_rate

    def update_weights(self, avg : np.ndarray[float]):
        self.__weights = np.add(self.__weights, avg)
    
    def update_bias(self, avg : float):
        self.__bias += avg

    def serialize(self): 
        numpyData = {
            "weights": self.__weights,
            "bias": self.__bias,
        }
        return numpyData
    
    def load(self, data):
        for i in range(len(data["weights"])):
            self.__weights = np.array(data["weights"])
        self.__bias = data["bias"]

    def __str__(self) -> str:
        s = "Weights : " + str(self.__weights) + "\n"
        s += "Bias : " + str(self.__bias) + "\n"
        return s