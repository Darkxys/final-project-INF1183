import numpy as np
class Layer:
  def __init__(self,prev_n, config=None):
    self.__activation = config.get_activation()
    self.__n = config.get_size()
    self.W = np.random.rand(self.__n, prev_n) - 0.5
    self.b = np.random.rand(self.__n, 1) - 0.5 
    self.Z = None 
    self.A = None 
    self.dZ = None 
    self.dW = None 
    self.db = None

  def forward_pass(self, X):
    self.Z = self.W.dot(X) + self.b 
    self.A = self.__activation.value[0](self.Z)
    return self.A
  
  def __one_hot(self, Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

  def backward_pass(self, m, prev_layer, next_layer, X, Y): 
    if next_layer is None:
      self.dZ = self.A - self.__one_hot(Y)
    else:
      self.dZ = next_layer.W.T.dot(next_layer.dZ) * self.__activation.value[1](self.Z)
    
    self.dW = 1 / m * self.dZ.dot((X if prev_layer is None else prev_layer.A).T)
    self.db = 1 / m * np.sum(self.dZ)
  
  def update_params(self, learning_rate):
    self.W -= learning_rate * self.dW
    self.b -= learning_rate * self.db  