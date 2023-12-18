from NN.layer_config import LayerConfig
from NN.layer import Layer
from NN.perceptron import Perceptron
from NN.exceptions.invalid_input_size_exception import InvalidInputSizeException
import numpy as np

import json
from json import JSONEncoder
from numpy_array_encoder import NumpyArrayEncoder

class NeuralNetwork:
    def __init__(self, layer_configs: np.ndarray[LayerConfig]) -> None:
        self.__layers : np.ndarray[Layer] = [Layer(
            layer_configs[i].get_size(), 
            layer_configs[i].get_activation(), 
            1 if i == 0 else layer_configs[i - 1].get_size(),
            True if i == 0 else False,
        ) for i in range(len(layer_configs))]
    
    def predict(self, inputs: np.ndarray[float]) -> np.ndarray[float]:
        self.evaluate(inputs)
        return self.__layers[len(self.__layers) - 1].get_layer_result()
    
    def evaluate(self, inputs: np.ndarray[float]) -> None:
        for i in range(len(self.__layers)): 
            if i == 0:
                self.__layers[i].input_pass(inputs)
            else:
                self.__layers[i].forward_pass(self.__layers[i - 1])

    def backward_pass(self, learning_rate) -> None:
        if len(self.__layers) == 1:
            return
        for i in reversed(range(1, len(self.__layers))):
            self.__layers[i].backward_pass(self.__layers[i - 1], learning_rate)
    
    def calculate_deltas(self, inputs: np.ndarray[float], outputs: np.ndarray[float]) -> None:
        for i in reversed(range(1, len(self.__layers))):
            if i == len(self.__layers) - 1:
                self.__layers[i].deltas_output_pass(outputs)
            else:
                self.__layers[i].deltas_pass(self.__layers[i + 1])

    def __train_step(self, inputs: np.ndarray[float], outputs: np.ndarray[float], learning_rate) -> None:
        self.evaluate(inputs)

        self.calculate_deltas(inputs, outputs)
        self.backward_pass(learning_rate)

    def __update_weights(self):
        for i in range(len(self.__layers)):
            if i > 0:
                self.__layers[i].update_weights()

    def train_batch(self, inputs: np.ndarray[np.ndarray[float]], outputs: np.ndarray[np.ndarray[float]], epoch = 100, learning_rate = 0.1, batch_size = 10, x_test_data = None, y_test_data = None) -> None:
        if len(inputs) != len(outputs):
            raise InvalidInputSizeException()
        for j in range(epoch):
            print("Epoch : #" + str(j) + "/" + str(epoch))
            for i in range(len(inputs)):
                if i % batch_size == 0 and i > 0: 
                    print(str(i) + " completed.")
                    self.__update_weights()
                self.__train_step(inputs[i], outputs[i], learning_rate)
                if i % 500 == 0 and i > 0: 
                    if (x_test_data is not None and y_test_data is not None): 
                        self.__internal_test(x_test_data[800:], y_test_data[800:])
            self.__update_weights()
            self.export_model("./models/model_" + str(j) + ".json")
            if (x_test_data is not None and y_test_data is not None): 
                self.__internal_test(x_test_data, y_test_data)

    def __internal_test(self, x_test_data, y_test_data):
        good = 0
        total = 0
        for i in range(len(x_test_data)):
            res = self.predict(x_test_data[i])
            max_index = np.argmax(res)
            if max_index == y_test_data[i]:
                good += 1
            total += 1

        acc = good / total
        print("Result : " + str(acc))

    def export_model(self, file_path = "./models/model.json") -> None:
        serialized_layers = []
        for layer in self.__layers:
            serialized_layers.append(layer.serialize())
        nn_data = {
            "layers": serialized_layers
        }
        with open(file_path, "w") as write_file:
            json.dump(nn_data, write_file, cls=NumpyArrayEncoder)
        print("Done writing neural network into file")

    def import_model(self, file_path = "./models/model.json") -> None:
        data = open(file_path,)
        json_data = json.load(data)
        for i in range(len(json_data["layers"])):
            self.__layers[i].load(json_data["layers"][i])
        print("Done loading neural network from file")
    
    def __str__(self) -> str:
        s = "Neural network : \n"
        for layer in self.__layers:
            s += str(layer)
        return s
        