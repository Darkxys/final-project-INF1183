from NN.perceptron import Perceptron
import numpy as np

class Layer: 
    def __init__(self, n, activation, prev_n = 1, is_input = False) -> None:
        self.__perceptrons : np.ndarray[Perceptron] = [Perceptron(prev_n, activation, is_input) for _ in range(n)]
        self.__deltas : np.ndarray[float] = []
        self.reset_memory()
    
    def reset_memory(self) -> None:
        self.__delta_avg : np.ndarray = []
        self.__delta_bias_avg : np.ndarray[float] = []
        self.__batch_counter : int = 0
        self.__layer_result : np.ndarray[float] = []

    def add_to_memory(self, result : np.ndarray[float]) -> None:
        self.__layer_result = result
        # self.__batch_counter += 1
        # self.__delta_avg += result

    def get_layer_result(self) -> np.ndarray[float]:
        return self.__layer_result
    
    def get_deltas(self) -> np.ndarray[float]:
        return self.__deltas
    
    def __predict(self, inputs : np.ndarray[float]) -> np.ndarray[float]:
        result = []
        for perceptron in self.__perceptrons:
            result.append(perceptron.predict(inputs))
        return result

    def input_pass(self, inputs : np.ndarray[float]) -> None:
        result = []
        for i in range(len(self.__perceptrons)):
            result.append(self.__perceptrons[i].predict([inputs[i]]))
        self.add_to_memory(result)

    def forward_pass(self, prev_layer) -> None:
        self.add_to_memory(self.__predict(prev_layer.get_layer_result()))

    def backward_pass(self, prev_layer, learning_rate) -> None:
        delta_avg = []
        bias_avg = []
        for i in range(len(self.__perceptrons)): 
            weight_deltas = np.zeros(len(self.__perceptrons[i].get_weights()))
            for j in range(len(self.__perceptrons[i].get_weights())): 
                weight_deltas[j] = learning_rate * prev_layer.get_layer_result()[j] * self.__deltas[i]
            delta_avg.append(weight_deltas)
            bias_avg.append(learning_rate * 1 * self.__deltas[i])
        delta_avg = np.array([np.array(d) for d in delta_avg])
        bias_avg = np.array(bias_avg)
        self.__batch_counter += 1
        if len(self.__delta_avg) == 0:
            self.__delta_avg = delta_avg
        else:
            self.__delta_avg = np.add(delta_avg, self.__delta_avg)
        if len(self.__delta_bias_avg) == 0:
            self.__delta_bias_avg = bias_avg
        else:
            self.__delta_bias_avg = np.add(bias_avg, self.__delta_bias_avg)
    
    def deltas_output_pass(self, outputs : np.ndarray[float]) -> None:
        self.__deltas = np.subtract(outputs, self.get_layer_result())
    
    def deltas_pass(self, next_layer) -> None: 
        self.__deltas = np.zeros(len(self.__perceptrons))
        for i in range(len(self.__perceptrons)):
            delta_sums = 0
            for j in range(len(next_layer.get_deltas())):
                mul = np.multiply(self.__perceptrons[i].get_weights(), next_layer.get_deltas()[j])
                delta_sums += np.sum(mul)
            
            self.__deltas[i] = self.__perceptrons[i].differentiate(self.__layer_result[i]) * delta_sums
    
    def update_weights(self) -> None: 
        for i in range(len(self.__perceptrons)):
            weights = self.__delta_avg[i]
            divs = np.empty(len(weights))
            divs.fill(self.__batch_counter)
            avg = np.divide(weights, divs)
            self.__perceptrons[i].update_weights(avg)

            bias = self.__delta_bias_avg[i]
            self.__perceptrons[i].update_bias(bias / self.__batch_counter)
        self.reset_memory()
    
    def serialize(self): 
        serialized_perceptrons = []
        for perceptron in self.__perceptrons:
            serialized_perceptrons.append(perceptron.serialize())
        layer_data = {
            "perceptrons": serialized_perceptrons
        }
        return layer_data
    
    def load(self, data):
        for i in range(len(data["perceptrons"])):
            self.__perceptrons[i].load(data["perceptrons"][i])

    def __str__(self) -> str:
        s = "Layer : \n"
        for perceptron in self.__perceptrons:
            s += str(perceptron)
        return s