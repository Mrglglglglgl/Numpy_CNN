import numpy as np
import mlp_tools as mlt
import cnn_tools as cnt

def pred(data, mlp_w_1, mlp_b_1, mlp_w_2, mlp_b_2, conv_w_1, conv_b_1, conv_w_2, conv_b_2):
  """
  Predicts the labels of a given dataset
  (essentially performs  forward pass)
  """
  conv_1_output, conv_1_bpass_step = cnt.conv_fpass(data, conv_w_1, conv_b_1)
  max_pool_1_output, max_pool_1_bpass_step = cnt.max_pooling_fpass(conv_1_output)
  conv_2_output, conv_2_bpass_step = cnt.conv_fpass(max_pool_1_output, conv_w_2, conv_b_2)
  max_pool_2_output, max_pool_2_bpass_step = cnt.max_pooling_fpass(conv_2_output)
  flattened_output = np.reshape(max_pool_2_output, [-1, 7*7*8])
  mlp_z_1, mlp_a_1, mlp_z_2, mlp_a_2 = mlt.mlp_fpass(flattened_output, mlp_w_1, mlp_b_1, mlp_w_2, mlp_b_2)

  return mlp_a_2

def measure_accuracy(predictions, labels):
  """
  Calculates the accuracy of the classifier by checking
  the indices of the two one-hot vectors
  """
  correct_predictions = np.equal(np.argmax(predictions,axis=1), np.argmax(labels,axis=1))
  accuracy = np.mean(correct_predictions*1.0)

  return accuracy