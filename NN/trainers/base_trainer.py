import numpy as np

class BaseTrainer: 
    def __init__(self, inputs : np.ndarray[np.ndarray[float]], outputs : np.ndarray[float]):
        self.__inputs : np.ndarray[np.ndarray[float]] = inputs
        self.__outputs : np.ndarray[float] = outputs

    def get_inputs(self):
        return self.__inputs
    
    def get_outputs(self):
        return self.__outputs