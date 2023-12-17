from NN.perceptron import Perceptron

class Layer: 
    def __init__(self, n, activation, prev_n = 1) -> None:
        self.__perceptrons : list[Perceptron] = [Perceptron(prev_n, activation) for _ in range(n)]
        self.__memory : list[float] = []
    
    def get_memory(self):
        return self.__memory
    
    def __predict(self, inputs : list[float]) -> list[float]:
        result = []
        for perceptron in self.__perceptrons:
            result.append(perceptron.predict(inputs))
        return result

    def input_pass(self, inputs : list[float]) -> None:
        self.__memory = []
        for i in range(len(self.__perceptrons)):
            self.__memory.append(self.__perceptrons[i].predict([inputs[i]]))

    def forward_pass(self, prev_layer) -> None:
        self.__memory = self.__predict(prev_layer.get_memory())
    
    def __str__(self) -> str:
        s = "Layer : \n"
        for perceptron in self.__perceptrons:
            s += str(perceptron)
        return s