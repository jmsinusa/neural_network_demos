'''
Project: neural_network_demos
Module name: scratch
Purpose:
Created: 18/11/2017 12:08
Author: james
'''


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.datasets import mnist
from keras.utils import to_categorical


def load_mnist():
    #Load MNIST data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    x_train_mean = np.mean(X_train)
    x_train_std = np.std(X_train)
    X_train = (X_train - x_train_mean) / x_train_std
    X_test = (X_test - x_train_mean) / x_train_std
    # one hot
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    return (X_train, Y_train), (X_test, Y_test)

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True,
          conv1_stride=1, conv2_stride=1):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param learning_rate:
    :param num_epochs:
    :param minibatch_size:
    :param print_cost:
    :return:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()

    (m, n_H0, n_W0, _) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost



    # Placeholder for input data
    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, 1))
    Y = tf.placeholder(tf.float32, shape = [None, n_y])
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters, conv1_stride = conv1_stride,
                             conv2_stride = conv2_stride)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9, beta2=0.999).minimize(cost)

    # Initialize all the variables globally

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            batch_indx = create_minibatch_index(m, minibatch_size)
            num_minibatches = int(m / minibatch_size)
            for batch_no in range(num_minibatches):
                (minibatch_X, minibatch_Y) = get_minibatch(X_train, Y_train, batch_indx, batch_no)
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters

def initialize_parameters():
    W1 = tf.get_variable("W1", [5, 5, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5, 5, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1,
                  "W2": W2}
    return parameters

def forward_propagation(X, parameters, conv1_stride = 2, conv2_stride = 2):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    :param X: input dataset placeholder
    :param parameters: dict containing W1 and W2
    :return: Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'

    Z1 = tf.nn.conv2d(X, W1, strides=[1, conv1_stride, conv1_stride, 1], padding='VALID')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 2x2, sride 2, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, conv2_stride, conv2_stride, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn=None)

    return Z3

def compute_cost(Z3, Y):
    """
    Compute cost
    :param Z3: LINEAR output of last layer
    :param Y: one hot labels
    :return: cost
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost

def create_minibatch_index(num_X, minibatch_size):
    """
    Create an index that can be used to produce random minibatches
    :param num_X: number of samples
    :param minibatch_size: number of minibatches
    :return:
    """
    assert type(num_X) == int
    assert type(minibatch_size) == int
    indx = np.array(range(num_X))
    batches = np.zeros(indx.shape)
    np.random.shuffle(indx)
    start = 0
    batch_no = 0
    while start < num_X:
        stop = start + minibatch_size
        if stop > num_X:
            stop = num_X
        batches[start:stop] = batch_no
        start = stop
        batch_no += 1
    return batches

def get_minibatch(X_train, Y_train, batches_indx, batch_no):
    """

    :param X_train:
    :param Y_train:
    :param batch_no:
    :return: mini_X_train, mini_Y_train
    """
    indxes = batches_indx == batch_no
    mini_X_train = X_train[indxes, :, :, :]
    mini_Y_train = Y_train[indxes, :]
    return mini_X_train, mini_Y_train

def visualise_filters(parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    for ii in range(W1.shape[3]):
        vals = W1[:, :, 0, ii]
        print vals
        plt.imshow(vals, cmap='gray')
        plt.show


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = load_mnist()
    #model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
    #      num_epochs=100, minibatch_size=64, print_cost=True)
    _, _, parameters = model(X_train, Y_train, X_test, Y_test,
                             learning_rate=0.006, num_epochs = 1,
                             conv1_stride=2, conv2_stride=2)
