import numpy as np
import helper_functions as hf


def conv_fpass(input_data, kernel_weights, kernel_biases, padding=1, strides=1):
  
  """
  Implements the forward pass of a convolutional layer
  
  Args: 
  input_data contains the output of the previous layer of dimension 
  (batch_size, channels, input height, input width)
  kernel_weights contains the weights of each kernel of the convolutional layer of 
  dimension (n_kernels, channels, kernel height, kernel width)
  kernel biases contains the biases (one for each kernel) of the convolutional layer (n_kernels, 1)
  padding is an integer, amount of padding around each image
  strides is an integer, controls how the kernels convolve around the input

  Outs:
  conv_output contains the output of the convolutional layer
  bpass_step contains information needed for the backward pass of the convolutional layer
  """
  
  # get the kernel weights and the input data dimensions, used to perform the convolution
  (n_kernels, channels, kernel_height, kernel_width) = kernel_weights.shape
  (batch_size, _ , input_data_height, input_data_width) = input_data.shape

  # compute the convolutional output dimensions
  conv_height = int((input_data_height - kernel_height + 2 * padding) / strides) + 1
  conv_width = int((input_data_width - kernel_width + 2 * padding) / strides) + 1
  conv_channels = channels
  
  # convert the data and kernel weights to 2-dimensional arrays
  input_data_matrix = hf.im2col(input_data, kernel_height, kernel_width,
                                     padding=padding, strides=strides)
  kernel_weights_matrix = np.reshape(kernel_weights, (n_kernels, -1))


  # apply forward propagation as in a fully connected layer
  conv_output = np.dot(kernel_weights_matrix,input_data_matrix) + kernel_biases
  
  # reshape the output back to its original 4-dimensional form to pass it to the next layer
  conv_output = np.reshape(conv_output,(n_kernels, conv_height, conv_width, batch_size))
  conv_output = np.transpose(conv_output, axes =(3, 0, 1, 2))
  
  # store the values needed for the backward pass of the convolutional layer
  bpass_step = (input_data, kernel_weights, kernel_biases, strides, padding, input_data_matrix)

  return conv_output, bpass_step


def max_pooling_fpass(input_data, kernel_size=2, strides=2):
  """
  Implements the forward pass for a max pooling layer
  
  Args: 
  input_data contains the output of the previous layer of dimension 
  (batch_size, channels, input_height, input_width)
  kernel_size is an integer, the height and width of the max_pooling "window"
  strides is an integer, controls how the kernel moves around the input

  Outs:
  maxpool_output contains the output of the max pooling layer
  bpass_step contains information needed for the backward pass of the max pooling layer
  """
  # get dimensions from the input_data
  (batch_size, channels , input_data_height, input_data_width) = input_data.shape
  
  # compute the dimensions of the output of the max pooling layer
  maxpool_height = int((input_data_height - kernel_size) / strides) + 1
  maxpool_width = int((input_data_width - kernel_size) / strides) + 1
  maxpool_channels = channels
  
  # reshape the input_data so that their "depth (channels)" is moved to the first dimension. This is done so that
  # after converting to 2d each maxpool window corresponds to a separate column for each channel
  input_data_matrix = np.reshape(input_data, (batch_size*channels, 1, input_data_height, input_data_width))
  
  # convert the input_data from 4-dimensional array to 2-dimensional array (each column corresponds to a 
  # single maxpool window of data --with the different channels of the same window corresponding to concecutive columns)
  input_data_matrix = hf.im2col(input_data_matrix, kernel_size, kernel_size, padding=0, strides=strides)
  
  # get the maximum elements per column (each column corresponds to a single maxpool window of data)
  maxpool_output_matrix = np.max(input_data_matrix, axis=0)
  
  # reshape the output to its original dimensions to feed it to the next layer
  maxpool_output = np.reshape(maxpool_output_matrix, (maxpool_height, maxpool_width, batch_size,
                                                      maxpool_channels))
  maxpool_output = np.transpose(maxpool_output, axes=(2, 3, 0, 1))
  
  # get the indices of the maximum element per column (needed for the bpass)
  max_indices = np.argmax(input_data_matrix, axis=0)
  
  # store the values needed for the backward pass
  bpass_step = (input_data, kernel_size, strides, input_data_matrix, max_indices)
  
  return maxpool_output, bpass_step

def max_pooling_bpass(dz_maxpool, bpass_step):
  """
  Implements the backward pass for a max pooling layer

  Args:

  dz_maxpool is the derivative of the loss w.r.t the output of the max pooling layer of
  dimensions (batch_size, channels, height, width)
  bpass_step contains the values needed for the backward pass of the convolutional layer
  that were stored during the forward pass

  Outs:
  maxpool_output contains the output of the max pooling layer
  """
  # unpack the backpropagation info computed during forward pass
  (input_data, pool_size, strides, input_data_matrix, max_indices) = bpass_step
  
  # get dimensions from the input_data
  (batch_size, channels , input_data_height, input_data_width) = input_data.shape
  
  # get the max pooling non-zero derivatives as a 1d array 
  dz_maxpool_flattened = dz_maxpool.transpose(2, 3, 0, 1)
  dz_maxpool_flattened = np.reshape(dz_maxpool_flattened, (-1,)) 
  
  # initialize the 2d form of the dloss/doutput of the next bpass layer with
  # zeros and then fill it with the non zero derivatives
  d_maxpool_input = np.zeros(shape=input_data_matrix.shape)
  d_maxpool_input[max_indices, np.arange(max_indices.shape[0])] = dz_maxpool_flattened
  
  # transform the 2d array into its 4d tensor form to pass it to the next layer
  d_maxpool_input = hf.col2im(d_maxpool_input, (batch_size * channels, 1, input_data_height, input_data_width),
                      pool_size, pool_size, padding=0, strides=strides)
  d_maxpool_input = np.reshape(d_maxpool_input,input_data.shape)
  
  return d_maxpool_input

def conv_bpass(dz_conv, bpass_step):
  """
  Implements the backward pass for a convolutional layer

  Args:
  dz_conv is the derivative of the loss w.r.t the output of the convolutional layer of
  dimensions (batch_size, channels, height, width)
  bpass_step contains the values needed for the backward pass of the convolutional layer
  that were stored during the forwardpass

  Outs:
  d_conv_input is the derivative of the loss w.r.t the input of the convolutional layer of
  dimensions (batch_size, channels, input_data_height, input_data_width)
  d_conv_w is the derivative of the loss w.r.t the kernel weights of the convolutional layer of
  dimensions (n_kernels, channels, kernel_height, kernel_width)
  d_conv_b is the derivative of the loss w.r.t the kernel biases of the convolutional layer of
  dimensions (n_kernels,1)
  """
  
  # get the kernel weights and the input data dimensions, used to perform the transpose convolution
  (input_data, kernel_weights, kernel_biases, strides, padding, input_data_matrix) = bpass_step
  (n_kernels, channels, kernel_height, kernel_width) = kernel_weights.shape

  # reshape the derivative of the loss w.r.t output to a 2-dimensional array (matrix)
  dz_conv_matrix = np.transpose(dz_conv, axes=(1, 2, 3, 0))
  dz_conv_matrix = np.reshape(dz_conv_matrix,(n_kernels, -1))

  # compute the derivative w.r.t the kernel weights as if it was a fully
  # connected layer
  d_conv_w = np.dot(dz_conv_matrix, np.transpose(input_data_matrix))

  # compute the derivative with respect to the biases of the convolutional layers
  # (one bias for each kernel) by summing across all dimensions except the 
  d_conv_b = np.reshape(np.sum(dz_conv, axis=(0, 2, 3)), (n_kernels, -1))

  # reshape the kernel weights to a 2 dimensional matrix
  kernel_weights_matrix = np.reshape(kernel_weights, (n_kernels, -1))

  # compute the derivative of the loss w.r.t the input of the convolutional layer as if
  # it was a fully connected layers
  d_conv_input_matrix = np.dot(np.transpose(kernel_weights_matrix),dz_conv_matrix)

  # reshape the derivative of the input to its 4-dimensional shape to pass it to the next stage
  # of the backward pass
  d_conv_input = hf.col2im(d_conv_input_matrix, input_data.shape, kernel_height, kernel_width,
                                padding=padding, strides=strides)
  
  # reshape the derivatives w.r.t the kernel weights to their original 4-dimensional shape to pass
  # it the to next stage of the backward pass
  d_conv_w = np.reshape(d_conv_w, kernel_weights.shape)
  
  return d_conv_input, d_conv_w, d_conv_b
