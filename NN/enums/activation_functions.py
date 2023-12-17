from enum import Enum

class Wrapper: 
    def __init__(self, function):
        self.__function = function

    def __call__(self, *args, **kwargs):
        return self.__function(*args, **kwargs)

def ReLu(input): 
    if input > 0:
        return input
    return 0

def Linear(input): 
    return input

class ActivationFunctions(Enum): 
    ReLu = Wrapper(ReLu)
    Linear = Wrapper(Linear)

