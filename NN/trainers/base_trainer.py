class BaseTrainer: 
    def __init__(self, inputs, outputs):
        self.__inputs = inputs
        self.__outputs = outputs

    def get_inputs(self):
        return self.__inputs
    
    def get_outputs(self):
        return self.__outputs