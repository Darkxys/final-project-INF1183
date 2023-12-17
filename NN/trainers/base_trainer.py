class BaseTrainer: 
    def __init__(self, inputs : list[list[float]], outputs : list[float]):
        self.__inputs : list[list[float]] = inputs
        self.__outputs : list[float] = outputs

    def get_inputs(self):
        return self.__inputs
    
    def get_outputs(self):
        return self.__outputs