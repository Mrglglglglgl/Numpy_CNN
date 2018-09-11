import helper_functions as hf
import activation_functions as af
import loss_functions as lf
import evaluation as ev
import mlp_tools as mlt
import cnn_tools as cnt
import numpy as np

def ConvNet(train_data, train_labels, test_data, test_labels, num_of_epochs, batch_size, learning_rate, log_period_samples):
  
    # counter used for evaluation
    counter = 0
    
    # initialize the weights using Xavier initialization (uniform) and the biases with zero
    mlp_w_1, mlp_b_1, mlp_w_2, mlp_b_2, conv_w_1, conv_b_1, conv_w_2, conv_b_2 = hf.initialize_weights()
    
    # get the epochs and create the accuracy and loss lists
    epochs = [epoch for epoch in range(num_of_epochs)]
    train_loss=[]
    test_loss=[]
    train_accuracy=[]
    test_accuracy=[]
    log_period_updates = int(log_period_samples / batch_size)
    for i in range(num_of_epochs):
        print('Running epoch number: ', i,' ...')
        
        #create the batches
        shuffled_data, shuffled_labels = hf.randomize(train_data, train_labels)
        batches = [shuffled_data[k:k+batch_size] for k in range(0, train_data.shape[0], batch_size)]
        batches_labels = [shuffled_labels[k:k+batch_size] for k in range(0, train_data.shape[0], batch_size)]
        for batch, batch_labels in zip (batches, batches_labels):
            counter+=1
            
            # forward pass
            conv_1_output, conv_1_bpass_step = cnt.conv_fpass(batch, conv_w_1, conv_b_1)
            max_pool_1_output, max_pool_1_bpass_step = cnt.max_pooling_fpass(conv_1_output)
            conv_2_output, conv_2_bpass_step = cnt.conv_fpass(max_pool_1_output, conv_w_2, conv_b_2)
            max_pool_2_output, max_pool_2_bpass_step = cnt.max_pooling_fpass(conv_2_output)
            flattened_output = np.reshape(max_pool_2_output, (-1,))
            flattened_output = np.reshape(flattened_output, (-1, 7*7*8))
            mlp_z_1, mlp_a_1, mlp_z_2, mlp_a_2 = mlt.mlp_fpass(flattened_output, mlp_w_1, mlp_b_1, mlp_w_2, mlp_b_2)
            
            # backward pass
            d_b_2, d_w_2, d_b_1, d_w_1, d_a_0 = mlt.mlp_bpass(flattened_output, batch_labels, mlp_w_1, mlp_z_1, mlp_a_1, mlp_w_2, mlp_z_2, mlp_a_2)
            d_a_0 = np.reshape(d_a_0, max_pool_2_output.shape)
            d_pool_2 = cnt.max_pooling_bpass(d_a_0, max_pool_2_bpass_step)
            d_conv_2, d_conv_w_2, d_conv_b_2 = cnt.conv_bpass(d_pool_2, conv_2_bpass_step)
            d_pool_1 = cnt.max_pooling_bpass(d_conv_2, max_pool_1_bpass_step)
            d_conv_1, d_conv_w_1, d_conv_b_1 = cnt.conv_bpass(d_pool_1, conv_1_bpass_step)
            
            # update
            conv_w_1 = conv_w_1 - learning_rate*d_conv_w_1
            conv_b_1 = conv_b_1 - learning_rate*d_conv_b_1
            conv_w_2 = conv_w_2 - learning_rate*d_conv_w_2
            conv_b_2 = conv_b_2 - learning_rate*d_conv_b_2
            mlp_w_1, mlp_b_1, mlp_w_2, mlp_b_2 = mlt.mlp_update_weights(mlp_w_1, mlp_b_1, d_w_1, d_b_1, mlp_w_2, mlp_b_2, d_w_2, d_b_2, learning_rate)
            
            # evaluate
            if counter % log_period_updates == 0:
                predictions = ev.pred(shuffled_data[:11000], mlp_w_1, mlp_b_1, mlp_w_2, mlp_b_2, conv_w_1, conv_b_1, conv_w_2, conv_b_2)
                test_predictions = ev.pred(test_data, mlp_w_1, mlp_b_1, mlp_w_2, mlp_b_2, conv_w_1, conv_b_1, conv_w_2, conv_b_2)
                
                # store the loss and the accuracy for train and 20% of test
                train_accuracy.append(ev.measure_accuracy(predictions, shuffled_labels[:11000]))
                test_accuracy.append(ev.measure_accuracy(test_predictions, test_labels))
                
    return test_accuracy, train_accuracy