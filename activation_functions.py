import numpy as np

def relu(x):
  """
  Calculates the relu activation function
  """

  return np.maximum(x, 0.0)

def d_relu(x):
  """
  Calculates the derivative of the relu
  """
  x[x > 0] = 1.0
  x[x == 0] = 0.0
  x[x < 0] = 0.0

  return x

def softmax(x):
  """
  Calculates the softmax activation function
  """
  a = np.exp(x - np.max(x, axis =1)[:, None])
  b = np.sum(np.exp(x-np.max(x,axis =1)[:, None]), axis = 1)

  return a / b[:, None]
