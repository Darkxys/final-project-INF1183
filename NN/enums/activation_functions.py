from enum import Enum
from collections.abc import Callable
import math

class Wrapper: 
    def __init__(self, function : Callable[[float], float]) -> None:
        self.__function = function

    def __call__(self, *args, **kwargs) -> float:
        return self.__function(*args, **kwargs)

def ReLu(input: float) -> float: 
    if input > 0:
        return input
    return 0

def Linear(input: float) -> float: 
    return input

def Logistic(input: float) -> float:
    return 1 / (1 + math.exp(-input))

class ActivationFunctions(Enum): 
    ReLu = Wrapper(ReLu)
    Linear = Wrapper(Linear)
    Logistic = Wrapper(Logistic)

