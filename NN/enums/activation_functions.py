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

def ReLuDif(input: float) -> float: 
    return input > 0

def Linear(input: float) -> float: 
    return input

def LinearDif(input: float) -> float: 
    return 1

def Logistic(input: float) -> float:
    return 1 / (1 + math.exp(-input))

def LogisticDif(input: float) -> float:
    return input * (1 - input)

class ActivationFunctions(Enum): 
    ReLu = [Wrapper(ReLu), Wrapper(ReLuDif)]
    Linear = [Wrapper(Linear), Wrapper(LinearDif)]
    Logistic = [Wrapper(Logistic), Wrapper(LogisticDif)]

