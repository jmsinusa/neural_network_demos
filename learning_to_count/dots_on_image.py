'''
Project: neural_network_demos
Module name: dots_on_image
Purpose:
Created: 08/12/2017 10:06
Author: james
'''


import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model



def create_images(n_images, n_dots, img_size, min_sep=2):
    """
    Create n_images, each of img_size, each containing the relevant number of dots,
    separated by min_sep pixels.
    :param n_images: integer number of images
    :param n_dots: numpy array of number of dots in each image.
    :param img_size: (x, y) size
    :param min_sep: Number of pixels between dots.
    :return: data, of shape (n_images, x, y), values 1 (dot) or 0 (no dot).
    """
    assert type(n_images) == int
    assert len(n_dots) == n_images
    assert len(img_size) == 2
    assert type(min_sep) == int
    x_train = np.zeros((n_images, img_size[0], img_size[1]), dtype=np.bool)
    for img in range(n_images):
        ndotspicked = 0
        n_its = 0
        dotcoords = [] #list of dot locations
        while ndotspicked < n_dots[img]:
            #pick a dot location
            x = np.random.randint(0, img_size[0], 1)
            y = np.random.randint(0, img_size[1], 1)
            goodval = True
            for (xx, yy) in dotcoords:
                if euclid_dist(x, y, xx, yy) < min_sep:
                    goodval = False
            if goodval:
                dotcoords.append((x, y))
                x_train[img, x, y] = 1
                ndotspicked += 1
            n_its += 1
            if n_its > 1000:
                raise NotImplementedError('Iterations have exceeded 1000 for img %i'
                                          '. Trying to create %i dots.'% (img, n_dots[img]))

    return x_train

def euclid_dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def plot_xtrain(xtrain, ytrain):
    showplots = np.random.randint(0, xtrain.shape[0], 16)
    #fig = plt.figure()
    for ii in range(16):
        plt.subplot(4, 4, ii + 1)
        datanum = showplots[ii]
        plt.imshow(xtrain[datanum, :], cmap='gray')
        plt.axis('off')
        plt.title('y=%i'%ytrain[datanum])
    plt.show()

def category_cnn1(xtrain, ytrain, epochs=10):
    Y_train = to_categorical(ytrain)
    nclasses = Y_train.shape[1]
    X_train = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
    cnn1 = model_cnn1(X_train.shape[1:], nclasses)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cnn1.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    #cnn1.summary()
    cnn1.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=64, verbose=2)
    return cnn1

def category_cnn2(xtrain, ytrain, epochs=10):
    Y_train = to_categorical(ytrain)
    nclasses = Y_train.shape[1]
    X_train = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
    cnn1 = model_cnn2(X_train.shape[1:], nclasses)
    adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cnn1.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    #cnn1.summary()
    cnn1.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=64, verbose=2)
    return cnn1

def category_cnn3(xtrain, ytrain, epochs=10):
    Y_train = to_categorical(ytrain)
    nclasses = Y_train.shape[1]
    X_train = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
    cnn1 = model_cnn3(X_train.shape[1:], nclasses)
    adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cnn1.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    #cnn1.summary()
    cnn1.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=64, verbose=2)
    return cnn1

def category_cnn3(xtrain, ytrain, epochs=10):
    Y_train = to_categorical(ytrain)
    nclasses = Y_train.shape[1]
    X_train = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
    cnn1 = model_cnn4(X_train.shape[1:], nclasses)
    adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cnn1.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    #cnn1.summary()
    cnn1.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=64, verbose=2)
    return cnn1

def category_cnn4(xtrain, ytrain, epochs=10):
    Y_train = to_categorical(ytrain)
    nclasses = Y_train.shape[1]
    X_train = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
    cnn1 = model_cnn4(X_train.shape[1:], nclasses)
    adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cnn1.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    #cnn1.summary()
    cnn1.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=64, verbose=2)
    return cnn1

def score_model(model, xtest, ytest):
    Y_test = to_categorical(ytest)
    X_test = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2], 1))
    preds = model.evaluate(x=X_test,y=Y_test)
    print "\nLoss = " + str(preds[0])
    print "Test Accuracy = " + str(preds[1])

def show_test_answers(model, xtest, ytest, savefilename=None,
                      plot_first_35=False):
    X_test = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2], 1))
    preds = model.predict(X_test)
    score = np.sum(np.argmax(preds, axis=1)==ytest)
    score = float(score) / float(len(ytest))
    print 'TEST SCORE: %6.4f%%'%(100.0*score)
    #print preds[0]
    if plot_first_35:
        showplots=range(35)
    else:
        showplots = np.random.randint(0, xtest.shape[0], 35)
    #fig = plt.figure()
    for ii in range(35):
        plt.subplot(5, 7, ii + 1)
        datanum = showplots[ii]
        plt.imshow(xtest[datanum, :], cmap='gray')
        plt.axis('off')
        guess = int(np.argmax(preds[datanum]))
        if guess == ytest[datanum]:
            cc = 'green'
        else:
            cc = 'red'
        plt.title('y=%i, $\hat{y}$=%i'%(ytest[datanum], guess), fontsize = 9, color=cc)
    plt.subplots_adjust(left=0.11, bottom=0.05, right=0.89, top=0.95, wspace=0.08, hspace=0.21)
    #plt.show()
    plt.savefig(savefilename, dpi=300)

def model_cnn1(input_shape, nclasses):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(shape=input_shape)
    X = Conv2D(1, (3, 3), strides=(1, 1), padding='same', name='conv0')(X_input)
    X = Activation('relu')(X)
    #X = Conv2D(1, (3, 3), strides=(1, 1), padding='valid', name='conv1')(X_input)
    #X = Activation('relu')(X)
    #X = AveragePooling2D((2, 2), strides=(2,2), name='avg_pool0')(X)
    X = Flatten()(X)
    #X = Dense(225, activation='relu', name='fc0', use_bias=False)(X)
    #X = Dense(225, activation='relu', name='fc0')(X)
    X = Dense(100, activation='relu', name='fc1')(X)
    X = Dense(nclasses, activation='sigmoid', name='fc2')(X)
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='cnn1')
    return model

def model_cnn2(input_shape, nclasses):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(shape=input_shape)
    X = Conv2D(1, (3, 3), strides=(1, 1), padding='same', name='conv0')(X_input)
    X = Activation('relu')(X)
    #X = Conv2D(1, (3, 3), strides=(1, 1), padding='valid', name='conv1')(X_input)
    #X = Activation('relu')(X)
    #X = AveragePooling2D((2, 2), strides=(2,2), name='avg_pool0')(X)
    X = Flatten()(X)
    #X = Dense(225, activation='relu', name='fc0', use_bias=False)(X)
    #X = Dense(225, activation='relu', name='fc0')(X)
    X = Dense(10, activation='relu', name='fc1')(X)
    X = Dense(nclasses, activation='sigmoid', name='fc2')(X)
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='cnn1')
    return model

def model_cnn3(input_shape, nclasses):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(shape=input_shape)
    X = Conv2D(1, (3, 3), strides=(1, 1), padding='same', name='conv0')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), strides=(2,2), name='avg_pool0')(X)
    X = Conv2D(10, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), strides=(2,2), name='avg_pool1')(X)
    X = Flatten()(X)
    #X = Dense(225, activation='relu', name='fc0', use_bias=False)(X)
    #X = Dense(225, activation='relu', name='fc0')(X)
    #X = Dense(10, activation='relu', name='fc1')(X)
    X = Dense(nclasses, activation='sigmoid', name='fc2')(X)
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='cnn1')
    return model

def model_cnn4(input_shape, nclasses):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(shape=input_shape)
    X = Conv2D(1, (3, 3), strides=(1, 1), padding='same', name='conv0')(X_input)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), strides=(2,2), name='avg_pool0')(X)
    X = Conv2D(10, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), strides=(2,2), name='avg_pool1')(X)
    X = Conv2D(5, (1, 1), strides=(1, 1), padding='same', name='conv2')(X)
    X = Activation('relu')(X)
    #X = AveragePooling2D((2, 2), strides=(2,2), name='avg_pool1')(X)
    X = Conv2D(5, (3, 3), strides=(1, 1), padding='same', name='conv3')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), strides=(2,2), name='avg_pool3')(X)

    X = Flatten()(X)
    #X = Dense(225, activation='relu', name='fc0', use_bias=False)(X)
    #X = Dense(225, activation='relu', name='fc0')(X)
    #X = Dense(10, activation='relu', name='fc1')(X)
    X = Dense(nclasses, activation='sigmoid', name='fc2')(X)
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='cnn1')
    return model

def visualise_weights(model, weight_no):
    ww = model.get_weights()
    conv0 = ww[weight_no]
    conv0 = np.array(conv0).squeeze()
    if len(conv0.shape) == 2:
        plt.imshow(conv0, cmap='gray')
        conv0_disp = conv0
        conv0_disp = conv0_disp / np.max(np.abs(conv0_disp))
        conv0_disp = 100.0 * conv0_disp
        for xx in range(conv0.shape[0]):
            for yy in range(conv0.shape[1]):
                val = conv0_disp[yy, xx]
                plt.text(xx, yy, "%2.0f"%val, fontsize=12, horizontalalignment='center', color='red')
        plt.axis('off')
        plt.show()
    # for ii in range(len(ww)):
    #     print 'LAYER %i'%ii
    #     print len(ww[ii])

if __name__ == "__main__":
    n_train = 12800
    n_test = 500
    n_classes_train = 5
    n_classes_test = 5
    savefilename=r'/Users/james/blog/20171208-nn_counting/x_epoch_cnn1_img1.png'
    #savefilename = r'/Users/james/blog/20171208-nn_counting/test.png'
    y_train = np.random.randint(1, n_classes_train+1, n_train)
    x_train = create_images(n_train, y_train, (26, 26), min_sep=3)
    y_test = np.random.randint(1, n_classes_test+1, n_test)
    x_test = create_images(n_test, y_test, (26, 26), min_sep=3)
    #plot_xtrain(x_train, y_train)
    model = category_cnn1(x_train, y_train, epochs=20)
    model.summary()
    score_model(model, x_test, y_test)
    #show_test_answers(model, x_test, y_test, savefilename=savefilename)
    visualise_weights(model, 0)
    #
    # n_classes_test = 10
    # savefilename=r'/Users/james/blog/20171208-nn_counting/12_epoch_img2.png'
    # y_test = np.random.randint(1, n_classes_test+1, n_test)
    # x_test = create_images(n_test, y_test, (26, 26), min_sep=3)
    # show_test_answers(model, x_test, y_test, savefilename=savefilename)
    #
    # n_classes_test = 50
    # savefilename=r'/Users/james/blog/20171208-nn_counting/12_epoch_img3.png'
    # y_test = np.random.randint(1, n_classes_test+1, n_test)
    # x_test = create_images(n_test, y_test, (26, 26), min_sep=3)
    # show_test_answers(model, x_test, y_test, savefilename=savefilename)

    # savefilename = r'/Users/james/blog/20171208-nn_counting/12_epoch_1to36.png'
    # y_test = np.array(range(1, 36))
    # x_test = create_images(len(y_test), y_test, (26, 26), min_sep=3)
    # show_test_answers(model, x_test, y_test, savefilename=savefilename, plot_first_35=True)
    #
    # n_classes_test = 5
    # savefilename=r'/Users/james/blog/20171208-nn_counting/12_epoch_img1_minsep2.png'
    # y_test = np.random.randint(1, n_classes_test+1, n_test)
    # x_test = create_images(n_test, y_test, (26, 26), min_sep=2)
    # show_test_answers(model, x_test, y_test, savefilename=savefilename)
    #
    # n_classes_test = 5
    # savefilename = r'/Users/james/blog/20171208-nn_counting/12_epoch_img1_minsep1.png'
    # y_test = np.random.randint(1, n_classes_test+1, n_test)
    # x_test = create_images(n_test, y_test, (26, 26), min_sep=1)
    # show_test_answers(model, x_test, y_test, savefilename=savefilename)