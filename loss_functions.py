import numpy as np

def softmax(y, t):
  """
  Calculates the Cross-Entropy Loss
  """
  loss = -np.sum(np.multiply(t,np.log(y)))

  return loss