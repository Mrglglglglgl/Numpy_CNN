import numpy as np

def initialize_weights():
  """
  Initializes the MLP weights using Xavier initialization (uniform)
  and the biases with zero
  """
  uni_range = np.sqrt(6.0 / (392+32))
  mlp_w_1 = np.random.uniform(-uni_range, uni_range, size=(392,32))
  mlp_b_1 = np.zeros(shape=(1,32))
  uni_range = np.sqrt(6.0 / (32+10))
  mlp_w_2 = np.random.uniform(-uni_range, uni_range, size=(32,10))
  mlp_b_2 = np.zeros(shape=(1,10))
  uni_range =np.sqrt(6.0 / (3*3))
  conv_w_1 = np.random.uniform(-uni_range, uni_range, size=(8,1,3,3))
  conv_b_1 = np.zeros(shape=(8,1))
  uni_range = np.sqrt(6.0 / (3*3*8))
  conv_w_2 = np.random.uniform(-uni_range, uni_range, size=(8,8,3,3))
  conv_b_2 = np.zeros(shape=(8,1))

  return mlp_w_1, mlp_b_1, mlp_w_2, mlp_b_2, conv_w_1, conv_b_1, conv_w_2, conv_b_2


def randomize(dataset, labels):
  """
  Shuffles data and their respective labels while
  maintaining the respective indices
  """
  permutation = np.random.permutation(dataset.shape[0])
  shuffled_a = dataset[permutation]
  shuffled_b = labels[permutation]

  return shuffled_a, shuffled_b


def zero_padding(data, padding):
	"""
	Pad with zeros all images of the data

	Args:
	data is a numpy array of shape (batch_size, channels, image height, image_width)
	padding is an integer, amount of padding around each image

	Outs:
	padded data is a numpy array shape 
	(batch_size, channels, image height + 2*padding, image wdith + 2*padding)
	"""

	padded_data = np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
	                     'constant', constant_values=0)

	return padded_data

def im2col(input_data, kernel_height=3, kernel_width=3, padding=1, strides=1):
    """ 
    Transforms a 4d tensor to 2d tensor (matrix) each column of which corresponding the sliding
    kernel window (kernel_height x kernel_width) -- different channels of the same image
    are concatenated in the same column.
    
    Based on Stanford's cs231 im2col (http://cs231n.stanford.edu/)
    
    Args: 
    input_data is a 4d tensor of dimension (batch_size, channels, input height, input width)
    kernel_height is an integer, the sliding window's height
    kernel_width is an integer, the sliding window's width
    padding is an integer, amount of padding around each image
    strides is an integer, controls how the window convolve around the input

    Outs:
    input_data_matrix is 2d tensor (matrix) each column of which corresponds to the sliding
    kernel window (kernel_height x kernel_width) at each time. Note: different channels of
    the same image are concatenated in the same column.
    """

    (_, channels, input_data_height, input_data_width) = input_data.shape
    
    output_height = int((input_data_height + 2 * padding - kernel_height) / strides + 1)
    output_width = int((input_data_width + 2 * padding - kernel_width) / strides + 1)
    channel_slice = np.arange(channels)
    channel_slice = np.repeat(channel_slice, kernel_height * kernel_width)
    channel_slice = np.reshape(channel_slice,(-1, 1))
    channel_slice = channel_slice.astype(int)
    height_slice_0 = np.arange(kernel_height)
    height_slice_0 = np.repeat(height_slice_0, kernel_width)
    height_slice_0 = np.tile(height_slice_0, channels)
    height_slice_1 = np.arange(output_height)
    height_slice_1 = strides * np.repeat(height_slice_1, output_width)
    height_slice = np.reshape(height_slice_0, (-1, 1)) + np.reshape(height_slice_1, (1, -1))
    height_slice = height_slice.astype(int)   
    width_slice_0 = np.arange(kernel_width)
    width_slice_0 = np.tile(width_slice_0, kernel_height * channels)
    width_slice_1 = np.arange(output_width)
    width_slice_1 = strides * np.tile(width_slice_1, output_height)    
    width_slice = np.reshape(width_slice_0,(-1, 1)) + np.reshape(width_slice_1, (1, -1))
    width_slice = width_slice.astype(int)
    padded_input = zero_padding(input_data, padding)
    input_data_matrix = padded_input[:, channel_slice, height_slice, width_slice]
    input_data_matrix = np.transpose(input_data_matrix, axes=(1, 2, 0))
    input_data_matrix = np.reshape(input_data_matrix, (kernel_height * kernel_width * channels, -1))
    
    return input_data_matrix


def col2im(input_data_matrix, input_data_shape, kernel_height=3, kernel_width=3, padding=1,
                   strides=1):
    """ 
    Transforms a 2d tensor (matrix) to 4d tensor: each column of the 2d tensor corresponds to the sliding
    kernel window (kernel_height x kernel_width) -- used during the backward pass of convolutional
    and maxpooling layers
    Based on Stanford's cs231 im2col (http://cs231n.stanford.edu/)
    Args: 
    input_data_matrix is a 2d matrix of dimensions:
    (kernel_height x kernel_width, batch_size xinput_data_height x input_data_width)
    kernel_height is an integer, the sliding window's height
    kernel_width is an integer, the sliding window's width
    padding is an integer, amount of padding around each image
    strides is an integer, controls how the window convolve around the input

    Outs:
    output_tensor is 4d tensor of dimensions same to the original images:
    (batch_size, channels, height, width)
    """
    (batch_size, channels, input_data_height, input_data_width) = input_data_shape
    output_height = int((input_data_height + 2 * padding - kernel_height) / strides + 1)
    output_width = int((input_data_width + 2 * padding - kernel_width) / strides + 1)
    channel_slice = np.arange(channels)
    channel_slice = np.repeat(channel_slice, kernel_height * kernel_width)
    channel_slice = np.reshape(channel_slice,(-1, 1))
    channel_slice = channel_slice.astype(int)
    height_slice_0 = np.arange(kernel_height)
    height_slice_0 = np.repeat(height_slice_0, kernel_width)
    height_slice_0 = np.tile(height_slice_0, channels)
    height_slice_1 = np.arange(output_height)
    height_slice_1 = strides * np.repeat(height_slice_1, output_width)
    height_slice = np.reshape(height_slice_0, (-1, 1)) + np.reshape(height_slice_1, (1, -1))
    height_slice = height_slice.astype(int)   
    width_slice_0 = np.arange(kernel_width)
    width_slice_0 = np.tile(width_slice_0, kernel_height * channels)
    width_slice_1 = np.arange(output_width)
    width_slice_1 = strides * np.tile(width_slice_1, output_height)    
    width_slice = np.reshape(width_slice_0,(-1, 1)) + np.reshape(width_slice_1, (1, -1))
    width_slice = width_slice.astype(int)  
    input_data_tensor = np.reshape(input_data_matrix, (channels * kernel_height * kernel_width, -1, batch_size))
    input_data_tensor = np.transpose(input_data_tensor, axes = (2, 0, 1))
    tensor_height = input_data_height + 2 * padding 
    tensor_width =  input_data_width + 2 * padding
    output_tensor = np.zeros((batch_size, channels, tensor_height, tensor_width))  
    np.add.at(output_tensor, (slice(None), channel_slice, height_slice, width_slice), input_data_tensor)
    if padding != 0:
        output_tensor = output_tensor[:, :, padding:-padding, padding:-padding]

    return output_tensor