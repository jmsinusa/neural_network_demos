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
    plt.imshow(dd.img)
    plt.show()


if __name__ == "__main__":
    create_trinagles()