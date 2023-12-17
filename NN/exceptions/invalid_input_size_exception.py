class InvalidInputSizeException(Exception): 
    def __init__(self, message = "Mismatch between the perceptron's weight array size and the provided input."):
        self.__message = message
        super().__init__(self.__message)