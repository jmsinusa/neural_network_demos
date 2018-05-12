'''
Project: neural_network_demos
Module name: dots_on_image_count
Purpose:
Created: 12/12/2017 19:23
Author: james
'''


import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam,SGD
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential


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

def create_images_quicker(n_images, n_dots, img_size):
    """
    Create n_images, each of img_size, each containing the relevant number of dots,
    No guaranteed separation.
    :param n_images: integer number of images
    :param n_dots: numpy array of number of dots in each image.
    :param img_size: (x, y) size
    :return: data, of shape (n_images, x, y), values 1 (dot) or 0 (no dot).
    """
    assert type(n_images) == int
    assert len(n_dots) == n_images
    assert len(img_size) == 2
    x_train = np.zeros((n_images, img_size[0], img_size[1]), dtype=np.bool)

    for img in range(n_images):
        ## put some random dots in
        dots_needed = n_dots[img]
        ndotspicked = 0
        while ndotspicked < n_dots[img]:
            x_train[img, :, :] = add_n_examples(dots_needed, x_train[img, :, :], img_size)
            ndotspicked = np.sum(x_train[img, :, :])
            dots_needed = n_dots[img] - ndotspicked

    return x_train

def add_n_examples(n, data, size):
    x = np.random.randint(0, size[0], n)
    y = np.random.randint(0, size[1], n)
    data[x, y] = 1
    return data

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

def cnn1(xtrain, ytrain, loss=None, cnn_model_no = 1, epochs=10, lr=0.001, beta_1=0.9, beta_2=0.999,
         decay=0.0):
    """
    Train network
    """
    Y_train = ytrain.astype(np.float)
    X_train = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
    if cnn_model_no == 1:
        cnn1 = model_cnn1(X_train.shape[1:])
    elif cnn_model_no == 2:
        cnn1 = model_cnn2(X_train.shape[1:])
    elif cnn_model_no == 3:
        cnn1 = model_cnn3(X_train.shape[1:])
    elif cnn_model_no == 4:
        cnn1 = model_cnn4(X_train.shape[1:])
    elif cnn_model_no == 5:
        cnn1 = model_cnn5(X_train.shape[1:])
    elif cnn_model_no == 6:
        cnn1 = model_cnn6(X_train.shape[1:])
    else:
        raise NotImplementedError()
    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=1e-08, decay=decay)
    sgd = SGD(lr=lr, momentum=beta_1)
    cnn1.summary()
    # checkpoint
    filepath = ("/Users/james/blog/20171208-nn_counting/4_dots_counting_v3/weights/"
                "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    #plot_model(cnn1, to_file=r'/Users/james/blog/20171208-nn_counting/2_dots_counting/cnn%i_model.png'%(cnn_model_no))
    #cnn1.compile(optimizer=adam, loss=loss, metrics=["accuracy"])
    cnn1.compile(optimizer=sgd, loss=loss, metrics=["accuracy"])
    history = cnn1.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=512, callbacks=callbacks_list, verbose=2)
    #history = cnn1.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=64, verbose=2)
    plot_loss(history)
    return cnn1


def plot_loss(history):
    acc = np.array(history.history['acc'])
    loss = np.array(history.history['loss'])
    epochs = np.array(range(len(loss)))
    # plt.plot(epochs, acc, label = 'acc')
    plt.plot(epochs, loss, label='loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(r'/Users/james/blog/20171208-nn_counting/4_dots_counting_v3/latestloss.png', dpi=300)
    plt.ylim([0, 300])
    plt.ylabel('Loss (clipped to 300)')
    plt.savefig(r'/Users/james/blog/20171208-nn_counting/4_dots_counting_v3/latestloss_max300.png', dpi=300)
    plt.ylim([0, 200])
    plt.ylabel('Loss (clipped to 200)')
    plt.savefig(r'/Users/james/blog/20171208-nn_counting/4_dots_counting_v3/latestloss_max200.png', dpi=300)
    plt.ylim([0, 20])
    plt.ylabel('Loss (clipped to 20)')
    plt.savefig(r'/Users/james/blog/20171208-nn_counting/4_dots_counting_v3/latestloss_max20.png', dpi=300)

def mse_loss_func(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true), axis=-1)
    return loss

def abs_loss_func(y_true, y_pred):
    loss = K.mean(K.abs(y_pred - y_true), axis=-1)
    return loss

def score_model(model, xtest, ytest):
    Y_test = ytest
    X_test = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2], 1))
    preds = model.evaluate(x=X_test,y=Y_test)
    print "\nLoss = " + str(preds[0])
    print "Test Accuracy = " + str(preds[1])

def show_test_answers(model, xtest, ytest, savefilename=None,
                      plot_first_35=False):
    X_test = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2], 1))
    preds = model.predict(X_test).squeeze()
    preds_int = np.rint(preds).astype(np.int)
    score = np.sum(preds_int == ytest)
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
        guess = preds_int[datanum]
        if guess == ytest[datanum]:
            cc = 'green'
        else:
            cc = 'red'
        plt.title('y=%i, $\hat{y}$=%i'%(ytest[datanum], guess), fontsize = 8, color=cc)
    plt.subplots_adjust(left=0.11, bottom=0.05, right=0.89, top=0.95, wspace=0.08, hspace=0.21)
    #plt.show()
    plt.savefig(savefilename, dpi=300)

def show_test_answers2(model, xtest, ytest, savefilename=None,
                      plot_first_35=False):
    X_test = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2], 1))
    preds = model.predict(X_test).squeeze()
    preds_int = np.rint(preds).astype(np.int)
    score = np.sum(preds_int == ytest)
    score = float(score) / float(len(ytest))
    print 'TEST SCORE: %6.4f%%'%(100.0*score)
    #print preds[0]
    if plot_first_35:
        showplots=range(28)
    else:
        showplots = np.random.randint(0, xtest.shape[0], 28)
    #fig = plt.figure()
    for ii in range(28):
        plt.subplot(4, 7, ii + 1)
        datanum = showplots[ii]
        plt.imshow(xtest[datanum, :], cmap='gray')
        plt.axis('off')
        guess = preds_int[datanum]
        if guess == ytest[datanum]:
            cc = 'green'
        else:
            cc = 'red'
        plt.title('y=%i\n$\hat{y}$=%i'%(ytest[datanum], guess), fontsize = 8, color=cc)
    plt.subplots_adjust(left=0.11, bottom=0.05, right=0.89, top=0.95, wspace=0.08, hspace=0.21)
    #plt.show()
    plt.savefig(savefilename, dpi=300)
    print "FIg saved to %s"%savefilename

def model_cnn1(input_shape):
    model = Sequential()
    model.add(Conv2D(1, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Flatten())
    model.add(Dense(1, activation='relu', name='fc0'))
    return model

def model_cnn2(input_shape):
    model = Sequential()
    model.add(Conv2D(3, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(3, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Conv2D(3, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Conv2D(3, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Flatten())
    model.add(Dense(10, activation='relu', name='fc0'))
    model.add(Dense(1, activation='relu', name='fc1'))
    return model

def model_cnn3(input_shape):
    model = Sequential()
    model.add(Conv2D(9, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(9, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Conv2D(9, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Conv2D(5, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', name='fc0'))
    model.add(Dense(10, activation='relu', name='fc1'))
    model.add(Dense(1, activation='relu', name='fc2'))
    return model


def model_cnn4(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    model = Sequential()
    model.add(Conv2D(1, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
                     padding='valid'))
    model.add(Flatten())
    model.add(Dense(1, activation='relu', name='fc0'))
    return model

def model_cnn5(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    model = Sequential()
    model = Sequential()
    model.add(Conv2D(1, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', input_shape=input_shape))
    model.add(AveragePooling2D((81,81), strides=(81,81), padding='valid'))
    # model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
    #                  padding='valid'))
    # model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
    #                  padding='valid'))
    # model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
    #                  padding='valid'))
    model.add(Flatten())
    model.add(Dense(1, activation='relu', name='fc0'))
    return model

def model_cnn6(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    model = Sequential()
    #model.add(Conv2D(1, (1, 1), strides=(1, 1), activation='relu',
    #                 padding='same', input_shape=input_shape))
    model.add(AveragePooling2D((81,81), strides=(81,81), padding='valid', input_shape=input_shape))
    # model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
    #                  padding='valid'))
    # model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
    #                  padding='valid'))
    # model.add(Conv2D(1, (3, 3), strides=(3, 3), activation='relu',
    #                  padding='valid'))
    model.add(Flatten())
    model.add(Dense(1, activation='relu', name='fc0'))
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
    n_train = 24000
    #n_train = 60
    n_test = 1000
    #n_test= 10
    n_classes_train = 6000
    n_classes_test = 6000
    cnn_model_no = 6
    img_dim = 81
    #loss_func = 'mean_absolute_percentage_error' #abs_loss_func or #mse_loss_func
    loss_func = mse_loss_func  # abs_loss_func or #mse_loss_func
    #loss_func_str = 'mean_absolute_percentage_error'  # abs_loss_func or #mse_loss_func
    loss_func_str = 'mse_loss_func'  # abs_loss_func or #mse_loss_func
    epochs = 20
    savefilename=(r'/Users/james/blog/20171208-nn_counting'
                  r'/4_dots_counting_v3/cnn%i_epochs'
                  '%i_%s_imgdim%i_img1.png'%(cnn_model_no, epochs, loss_func_str, img_dim))
    #savefilename = r'/Users/james/blog/20171208-nn_counting/test.png'
    #y_train = np.random.randint(1, n_classes_train+1, n_train)
    y_train = np.random.randint(0, n_classes_train, n_train)
    x_train = create_images_quicker(n_train, y_train, (img_dim, img_dim))
    y_test = np.random.randint(0, n_classes_test, n_test)
    x_test = create_images_quicker(n_test, y_test, (img_dim, img_dim))
    #plot_xtrain(x_train, y_train)
    model = cnn1(x_train, y_train, loss=loss_func, cnn_model_no=cnn_model_no, epochs=epochs,
                 lr=0.1, beta_1=0.0, decay=0.000001)
    score_model(model, x_test, y_test)
    show_test_answers2(model, x_test, y_test, savefilename=savefilename)
    #visualise_weights(model, 0)

    savefilename = (r'/Users/james/blog/20171208-nn_counting/4_dots_counting_v3/cnn%i_epochs'
                    '%i_%s_imgdim%i_test1.png'%(cnn_model_no, epochs, loss_func_str, img_dim))
    y_test = 100*np.array(range(1, 29))
    x_test = create_images_quicker(len(y_test), y_test, (img_dim, img_dim))
    show_test_answers2(model, x_test, y_test, savefilename=savefilename, plot_first_35=True)

    savefilename = (r'/Users/james/blog/20171208-nn_counting/4_dots_counting_v3/cnn%i_epochs'
                   '%i_%s_imgdim%i_test2.png'%(cnn_model_no, epochs, loss_func_str, img_dim))
    y_test = 200*np.array(range(28))
    x_test = create_images_quicker(len(y_test), y_test, (img_dim, img_dim))
    show_test_answers2(model, x_test, y_test, savefilename=savefilename, plot_first_35=True)