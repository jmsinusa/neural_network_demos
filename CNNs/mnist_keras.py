'''
Project: neural_network_demos
Module name: scratch2
Purpose:
Created: 20/11/2017 17:22
Author: james
'''


import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input



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

def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(shape = input_shape)
    X = Conv2D(8, (5, 5), strides=(1, 1), padding='same', name='conv0')(X_input)
    ##X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    X = Conv2D(16, (5, 5), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    X = Flatten()(X)
    X = Dense(200, activation='relu', name='fc0')(X)
    X = Dense(100, activation='relu', name='fc1')(X)
    X = Dense(10, activation='sigmoid', name='fc2')(X)
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='NMIST_mod1')
    return model

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = load_mnist()
    print X_train.shape
    nmist = model(X_train.shape[1:])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nmist.compile(optimizer=adam, loss = "categorical_crossentropy", metrics = ["accuracy"])
    nmist.summary()
    nmist.fit(x = X_train, y = Y_train, epochs=1, batch_size = 64)

    preds = nmist.evaluate(x=X_test,y=Y_test)
    print "\nLoss = " + str(preds[0])
    print "Test Accuracy = " + str(preds[1])
