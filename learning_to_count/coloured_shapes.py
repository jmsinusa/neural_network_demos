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
import os
import pickle
# from keras.utils import to_categorical, plot_model
# from keras.optimizers import Adam,SGD
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
# from keras.models import Model
# from keras import backend as K
# from keras.callbacks import ModelCheckpoint
#
# from keras.models import Sequential

from data_creator import DataCreator


def create_square(n_forlabel, save_filename, noise_scale=30,
                  labelcolour=(230, 30, 30), labelsize=(3, 3), bkgdsize=(90, 90, 3)):
    '''Description'''
    dd = DataCreator(dtype=np.uint8)
    dd.zeros_bkgd(bkgdsize)
    # dd.add_noise(scale=10.0, abs_noise=True)
    errors = 0
    n_forlabel = n_forlabel
    for ii in range(n_forlabel):
        # errors += dd.add_shape(shape='square', colour=(255, 255, 255), size=(3, 3))
        errors += dd.add_shape(shape='square', colour=labelcolour, size=labelsize)
        # error = dd.add_shape(shape='square', colour=(10, 0, 255), size=(4, 1))

    dd.add_noise(scale=noise_scale, noise='uniform', abs_noise=True)
    # plt.imshow(dd.img)
    # plt.show()
    n_forlabel -= errors
    plt.imsave(arr=dd.img, fname=save_filename, format='png')
    return n_forlabel


def create_ex1(n_examples, save_dir):
    labels = {}
    n_forlabels = np.random.randint(0, high=50, size=n_examples)
    for eg_no in range(n_examples):
        save_filename = "sample%05i.png" % eg_no
        save_filepath = os.path.join(save_dir, save_filename)
        n_forlabel = n_forlabels[eg_no]
        n_label = create_square(n_forlabel, save_filepath)
        labels[save_filename] = n_label
    pickle.dump(labels, open(os.path.join(save_dir, 'labels.pkl'), "wb"))


def create_npy_dataset(dir_):
    assert os.path.isdir(dir_)
    pickle_file = os.path.join(dir_, 'labels.pkl')
    assert os.path.isfile(pickle_file)
    labels = pickle.load(open(pickle_file, "rb"))
    list_files_raw = os.listdir(dir_)
    png_list = []
    for ff in list_files_raw:
        if os.path.splitext(ff)[1] in ['.png']:
            png_list.append(ff)
    print("%i PNGs in directory." % len(png_list))
    # Load first img
    img0 = plt.imread(os.path.join(dir_, png_list[0]))
    img_shape_xy = img0.shape[:2]
    data_npy = np.zeros((len(png_list), img_shape_xy[0], img_shape_xy[1], 3), dtype=np.float32)
    labels_npy = np.zeros(len(png_list), dtype=np.uint8)
    for ii, png_ff in enumerate(png_list):
        img = plt.imread(os.path.join(dir_, png_ff))
        label = labels[png_ff]
        data_npy[ii, :, :, :] = img[:, :, :3]
        labels_npy[ii] = label
    np.save(os.path.join(dir_, 'data.npy'), data_npy)
    np.save(os.path.join(dir_, 'labels.npy'), labels_npy)

if __name__ == "__main__":
    dir_ = r'/Users/james/blog/20180512-ColouredShapeCounting/ex1'
    # create_ex1(10, r'/Users/james/blog/20180512-ColouredShapeCounting/ex1')
    create_npy_dataset(dir_)
    # data = np.load(os.path.join(dir_, 'data.npy'))
    # labels = np.load(os.path.join(dir_, 'labels.npy'))
    # print(labels[2])
    # plt.imshow(data[2, :, :, :])
    # plt.show()