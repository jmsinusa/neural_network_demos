"""
Project: neural_network_demos
Module name: data_creator
Purpose: Create toy counting problems.
Created: 28/12/2017 15:53
Author: james
"""

from __future__ import print_function

import numpy as np


class DataCreator(object):
    """
    Generic data creation class
    """
    def __init__(self, dtype=np.uint8):
        """
        None
        """
        self.img = None
        self.dtype = dtype
        self.mask = None


    def zeros_bkgd(self, size):
        """
        Create a numpy array of zeros.
        :param size: (x, y) in pixels
        :param dtype: int | float
        :return: numpy array
        """
        self.img =  np.zeros(size, dtype=self.dtype)
        self.mask = np.zeros_like(self.img, dtype=np.bool)


    def add_noise(self, noise='normal', mean=0, scale=1.0, abs_noise=True):
        """
        Add noise to bkgd
        :param noise: normal
        :param mean: mean of noise
        :param scale: scale (std for normal)
        :param abs_noise: Force all noise to be positive?
        :param dtype: int | float
        :return: bkgd + noise, a numpy array.
        """
        if noise == 'normal':
            noise_array = np.random.normal(loc=mean, scale=scale, size=self.img.shape)
        else:
            raise NotImplementedError('Noise type "%s" not recognised.'% noise)
        if abs_noise:
            noise_array = np.abs(noise_array, out=noise_array)
        out = self.img + noise_array
        self.img = out.astype(self.dtype)

    def add_shape(self, loc, shape = 'square', colour=(255, 255, 255), size = (3, 3)):
        """Add shape of self.img"""



if __name__ == "__main__":
    pass
