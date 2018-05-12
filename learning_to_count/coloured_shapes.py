'''
Project: neural_network_demos
Module name: coloured_shapes
Purpose:
Created: 11/05/2018 20:46
Author: james
'''

from __future__ import print_function

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

from data_creator import DataCreator


def create_trinagles():
    '''Description'''
    dd = DataCreator(dtype=np.uint8)
    dd.zeros_bkgd((100, 100, 3))
    # dd.add_noise(scale=10.0, abs_noise=True)
    for ii in range(10):
        error = dd.add_shape(shape='square', colour=(255, 255, 255), size=(3, 3))
        error = dd.add_shape(shape='square', colour=(255, 0, 0), size=(3, 3))
        error = dd.add_shape(shape='square', colour=(10, 0, 255), size=(4, 1))
    dd.add_noise(scale=40.0, abs_noise=True)
    plt.imshow(dd.img)
    plt.show()


if __name__ == "__main__":
    create_trinagles()