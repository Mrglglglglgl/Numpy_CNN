import numpy as np
import activation_functions as af


def mlp_fpass(data, w_1, b_1, w_2, b_2):
  """
  Initializes the MLP weights using Xavier initialization (uniform)
  and the biases with zero
  """    
  z_1 = np.add(np.dot(data, w_1), b_1)
  a_1 = af.relu(z_1)
  z_2 = np.add(np.dot(a_1, w_2), b_2)
  a_2 = af.softmax(z_2)

  return z_1, a_1, z_2, a_2

def mlp_bpass(data, labels, w_1, z_1, a_1, w_2, z_2, a_2):
  """
  Implements the MLP backward pass of the training process:
  Computes the derivatives of the cost function with respect
  to bias, weights and the data
  """
  d_z_2 = np.subtract(a_2, labels)
  d_b_2 = d_z_2
  d_w_2 = np.dot(np.transpose(a_1), d_z_2)
  d_a_1 = np.dot(d_z_2, np.transpose(w_2))
  d_z_1 = np.multiply(d_a_1, af.d_relu(z_1))
  d_b_1 = d_z_1
  d_w_1 = np.dot(np.transpose(data), d_z_1)
  d_a_0 = np.dot(d_z_1, np.transpose(w_1))

  return d_b_2, d_w_2, d_b_1, d_w_1, d_a_0

def mlp_update_weights(w_1, b_1, d_w_1, d_b_1, w_2, b_2, d_w_2, d_b_2, learning_rate):
  """
  Updates the weights of the MLP
  """
  new_w_1 = np.subtract(w_1, np.multiply(learning_rate, d_w_1))
  new_b_1 = np.subtract(b_1, np.multiply(learning_rate, np.sum(d_b_1, axis=0)))
  new_w_2 = np.subtract(w_2, np.multiply(learning_rate, d_w_2))
  new_b_2 = np.subtract(b_2, np.multiply(learning_rate, np.sum(d_b_2, axis=0)))

  return new_w_1, new_b_1, new_w_2, new_b_2