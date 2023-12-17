from NN.exceptions.invalid_input_size_exception import InvalidInputSizeException
from NN.enums.activation_functions import ActivationFunctions
import random

LOW = -1.0
HIGH = 1.0

class Perceptron: 
    def __init__(self, n: int, activation: ActivationFunctions) -> None:
        self.__weights : list[float] = [random.uniform(LOW, HIGH) for _ in range(n)]
        self.__bias : float = random.uniform(LOW, HIGH)
        self.__activation : ActivationFunctions = activation


    def diff(self, output : float, expected : float) -> float:
        return 2 * (expected - output)
    
    def activate(self, input : float) -> float:
        return self.__activation.value(input)
    
    def predict(self, inputs : list[float]) -> float: 
        print(str(inputs))
        print(str(self.__weights))
        if len(inputs) != len(self.__weights):
            raise InvalidInputSizeException()
        
        result : float = self.__bias
        for i in range(len(self.__weights)):
            result += self.__weights[i] * inputs[i]

        return self.activate(result)

    def evaluate(self, inputs : list[list[float]], outputs : list[float]) -> float:
        if len(outputs) == 0:
            return 0
        
        loss : float = 0
        for i in range(len(self.__weights)):
            o : float = self.predict(inputs[i])
            loss += self.diff(o, outputs[i])

        return abs(loss / len(outputs))
    
    def train(self, inputs : list[list[float]], outputs : list[float], learning_rate : float = 0.001) -> None:
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

    def __str__(self) -> str:
        s = "Weights : " + str(self.__weights) + "\n"
        s += "Bias : " + str(self.__bias) + "\n"
        return s