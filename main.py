from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from convnet import ConvNet

log_period_samples = 20000
batch_size = 100
experiments_list = []
settings = [(5, 0.01), (10, 0.001), (20, 0.001)]

def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

def training(settings):
    print('Training Model')
    for (num_epochs, learning_rate) in settings:
        mnist = get_data()
        eval_mnist = get_data()
        x_train = np.reshape(mnist.train.images ,[-1,1,28,28])
        y_train = mnist.train.labels
        x_test = np.reshape(mnist.test.images, [-1,1,28,28])
        y_test = mnist.test.labels
        test_accuracy, train_accuracy = ConvNet(x_train,
        y_train, x_test, y_test ,num_epochs, 
                                     batch_size, learning_rate, log_period_samples)
        experiments_list.append(
          ((num_epochs, learning_rate), train_accuracy, test_accuracy))
	print(experiments_list)

if __name__ == '__main__':
    training(settings)

