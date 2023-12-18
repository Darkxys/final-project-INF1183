from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
from KNN import KNN

import json
from json import JSONEncoder
from numpy_array_encoder import NumpyArrayEncoder

from NN.trainers.base_trainer import BaseTrainer
from NN.perceptron import Perceptron
from NN.enums.activation_functions import ActivationFunctions

from NN.neural_network import NeuralNetwork
from NN.layer_config import LayerConfig

mnist = fetch_openml(name='mnist_784', parser='auto')

def test_knn():
    index_number= np.random.permutation(70000)
    x1,y1=mnist.data.loc[index_number],mnist.target.loc[index_number]
    x1.reset_index(drop=True,inplace=True)
    y1.reset_index(drop=True,inplace=True)
    x_train , x_test = x1[:55000], x1[55000:]
    y_train , y_test = y1[:55000], y1[55000:]

    knn = KNN(5)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)

# test_knn()
    
def test_nn():
    index_number= np.random.permutation(10000)
    x1,y1=mnist.data.loc[index_number],mnist.target.loc[index_number]
    x1.reset_index(drop=True,inplace=True)
    y1.reset_index(drop=True,inplace=True)
    x_train , x_test = x1[:9000], x1[9000:]
    y_train , y_test = y1[:9000], y1[9000:]

    np_x_train = x_train.to_numpy().astype(float)
    np_y_train = y_train.to_numpy().astype(int)
    np_x_test = x_test.to_numpy().astype(float)
    np_y_test = y_test.to_numpy().astype(int)
    norm_x_train = (np_x_train + 1) / 256
    norm_x_test = (np_x_test + 1) / 256

    processed_y_train = [np.eye(10)[index] for index in np_y_train]
    processed_y_train = np.array(processed_y_train)

    configs = [
        LayerConfig(784, ActivationFunctions.Linear),
        # LayerConfig(32, ActivationFunctions.Logistic),
        LayerConfig(10, ActivationFunctions.Logistic),
    ]

    nn = NeuralNetwork(configs)

    nn.train_batch(
        norm_x_train, 
        processed_y_train, 
        x_test_data=norm_x_test,
        y_test_data=np_y_test,
        epoch=1000,
        learning_rate=0.1, 
        batch_size=10
    )

    good = 0
    total = 0
    for i in range(len(norm_x_test)):
        res = nn.predict(norm_x_test[i])
        max_index = np.argmax(res)
        if max_index == np_y_test[i]:
            good += 1
        total += 1

    acc = good / total
    print("Result : " + str(acc))

    nn.export_model()
    
test_nn()