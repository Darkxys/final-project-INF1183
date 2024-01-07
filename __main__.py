from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
from KNN import KNN
from NN.enums.layer_types import LayerTypes
import pandas as pd

import json
from json import JSONEncoder
from numpy_array_encoder import NumpyArrayEncoder
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
    data = pd.read_csv('./data/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.
    
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _,m_train = X_train.shape

    configs = [
        LayerConfig(784),
        LayerConfig(n=20, activation=ActivationFunctions.Logistic),
        LayerConfig(n=10, activation=ActivationFunctions.ReLu),
        LayerConfig(n=10, activation=ActivationFunctions.Softmax),
    ]

    nn = NeuralNetwork(configs)

    nn.training(
        m=m,
        X=X_train,
        Y=Y_train, 
        X_test=X_dev,
        Y_test=Y_dev,
        epoch=2000,
        learning_rate=0.1,
    )
    
test_nn()