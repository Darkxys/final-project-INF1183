from NN.layer import Layer
from NN.layer_config import LayerConfig
import numpy as np


class NeuralNetwork:
    def __init__(self, configs: LayerConfig):
        self.__layers = [
            Layer(config=configs[i], prev_n=configs[i - 1].get_size())
            for i in range(1, len(configs))
        ]

    def forward_prop(self, X):
        input = X
        for layer in self.__layers:
            input = layer.forward_pass(input)
        return input

    def backward_prop(self, m, X, Y):
        for i, layer in enumerate(reversed(self.__layers)):
            index = len(self.__layers) - i - 1
            prev_layer = None if index <= 0 else self.__layers[index - 1]
            next_layer = (
                None if index == len(self.__layers) - 1 else self.__layers[index + 1]
            )
            layer.backward_pass(
                m=m, X=X, Y=Y, prev_layer=prev_layer, next_layer=next_layer
            )

    def update_params(self, learning_rate):
        for layer in self.__layers:
            layer.update_params(learning_rate=learning_rate)

    def training(self, m, learning_rate, epoch, X, Y, X_test, Y_test):
        for i in range(epoch):
            self.forward_prop(X)
            self.backward_prop(m, X, Y)
            self.update_params(learning_rate)
            if i % 10 == 0:
                print("Epoch: ", i)
                self.test(X_test, Y_test)

    def test(self, X, Y):
        predictions = np.argmax(self.forward_prop(X), 0)
        print(np.sum(predictions == Y) / Y.size)
