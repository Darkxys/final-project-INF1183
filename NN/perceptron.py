from NN.exceptions.invalid_input_size_exception import InvalidInputSizeException
import random

LOW = -10.0
HIGH = 10.0

class Perceptron: 
    def __init__(self, n, activation):
        self.__weights = [random.uniform(LOW, HIGH) for _ in range(n)]
        self.__bias = random.uniform(LOW, HIGH)
        self.__activation = activation


    def diff(self, output, expected):
        return 2 * (expected - output)
    
    def activate(self, input):
        return input
    
    def predict(self, inputs): 
        if len(inputs) != len(self.__weights):
            raise InvalidInputSizeException()
        
        result = self.__bias
        for i in range(len(self.__weights)):
            result += self.__weights[i] * inputs[i]

        return result

    def evaluate(self, inputs, outputs):
        if len(outputs) == 0:
            return 0
        
        loss = 0
        for i in range(len(self.__weights)):
            o = self.predict(inputs[i])
            loss += self.diff(o, outputs[i])

        return abs(loss / len(outputs))
    
    def train(self, inputs, outputs, learning_rate = 0.001):
        if len(inputs) != len(outputs):
            raise InvalidInputSizeException()

        for i in range(len(inputs)):
            sum_of_inputs = 1
            for input in inputs[i]:
                sum_of_inputs += input
            
            prediction = self.predict(inputs[i])
            diff = abs(self.diff(prediction, outputs[i]))
            direction = 1
            if prediction > outputs[i]:
                direction = -1
            
            for j in range(len(self.__weights)):
                self.__weights[j] += direction * diff * (abs(inputs[i][j]) / sum_of_inputs) * learning_rate

            self.__bias += direction * diff * (1 / sum_of_inputs) * learning_rate

    def __str__(self):
        s = "Weights : " + str(self.__weights) + "\n"
        s += "Bias : " + str(self.__bias) + "\n"
        return s