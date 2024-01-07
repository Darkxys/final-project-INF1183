from enum import Enum
from collections.abc import Callable
import math
import numpy as np

class Wrapper: 
    def __init__(self, function : Callable[[float], float]) -> None:
        self.__function = function

    def __call__(self, *args, **kwargs) -> float:
        return self.__function(*args, **kwargs)

def ReLu(input: np.ndarray) -> np.ndarray: 
    return np.maximum(0, input)

def ReLuDif(input: np.ndarray) -> np.ndarray: 
    return input > 0

def LeakyReLu(input: np.ndarray) -> np.ndarray: 
    return np.maximum(input*0.01, input)

def LeakyReLuDif(input: np.ndarray) -> np.ndarray: 
    dx = np.ones_like(input)
    dx[input < 0] = 0.01
    return dx

def Softmax(input: np.ndarray) -> np.ndarray:
    exps = np.exp(input)
    return exps / sum(exps)

def Linear(input: np.ndarray) -> np.ndarray: 
    return input

def LinearDif(input: np.ndarray) -> np.ndarray: 
    return 1

def Logistic(input: np.ndarray) -> np.ndarray:
    return np.where(input < 0, np.exp(input) / (1 + np.exp(input)), 1 / (1 + np.exp(-input)))

def LogisticDif(input: np.ndarray) -> np.ndarray:
    return Logistic(input) * (1 - Logistic(input))

class ActivationFunctions(Enum): 
    ReLu = [Wrapper(ReLu), Wrapper(ReLuDif)]
    LeakyReLu = [Wrapper(LeakyReLu), Wrapper(LeakyReLuDif)]
    Linear = [Wrapper(Linear), Wrapper(LinearDif)]
    Logistic = [Wrapper(Logistic), Wrapper(LogisticDif)]
    Softmax = [Wrapper(Softmax), None]
