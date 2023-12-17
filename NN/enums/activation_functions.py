from enum import Enum
from collections.abc import Callable

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

class ActivationFunctions(Enum): 
    ReLu = Wrapper(ReLu)
    Linear = Wrapper(Linear)

