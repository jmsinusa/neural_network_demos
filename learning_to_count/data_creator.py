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
        self.img = np.zeros(size, dtype=self.dtype)
        self.mask = np.zeros((size[0], size[1]), dtype=np.bool)

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
        elif noise == 'uniform':
            noise_array = np.random.uniform(low=int(mean - scale), high=int(mean + scale),
                                            size=self.img.shape)
        else:
            raise NotImplementedError('Noise type "%s" not recognised.' % noise)
        if abs_noise:
            noise_array = np.abs(noise_array, out=noise_array)
        out = self.img + noise_array
        out[out > 255] = 255
        out[out < 0] = 0
        self.img = out.astype(self.dtype)

    def add_shape(self, shape='square', colour=(255, 255, 255), size=(3, 3)):
        """Add shape to random location in self.img, unless self.mask = True for any pixel."""
        (maxy, maxx, bands) = self.img.shape
        assert len(colour) == bands
        loc = None
        size_to_check_y = int(np.ceil((size[0] - 1.0) / 2.0)) + 1
        size_to_check_x = int(np.ceil((size[1] - 1.0) / 2.0)) + 1
        size_to_colour_y = int(np.ceil((size[0] - 1.0) / 2.0))
        size_to_colour_x = int(np.ceil((size[1] - 1.0) / 2.0))
        check_coords = []
        colour_coords = []
        attempts = 0
        while not loc:
            loc_y = np.random.randint(size_to_check_y, maxy - size_to_check_y)
            loc_x = np.random.randint(size_to_check_x, maxx - size_to_check_x)
            for ss in range(-size_to_check_y, size_to_check_y + 1):
                test_loc_y = loc_y + ss
                for tt in range(-size_to_check_x, size_to_check_x + 1):
                    test_loc_x = loc_x + tt
                    check_coords.append((test_loc_x, test_loc_y))
            for ss in range(-size_to_colour_y, size_to_colour_y + 1):
                test_loc_y = loc_y + ss
                for tt in range(-size_to_colour_x, size_to_colour_x + 1):
                    test_loc_x = loc_x + tt
                    colour_coords.append((test_loc_x, test_loc_y))

            all_false = True
            for cc_x, cc_y in check_coords:
                if self.mask[cc_y, cc_x]:
                    all_false = False
            if all_false:
                # Set mask to True and replace self.img with colour
                for cc_x, cc_y in check_coords:
                    self.mask[cc_y, cc_x] = True
                for cc_x, cc_y in colour_coords:
                    self.img[cc_y, cc_x, :] = colour
                loc = True
            else:
                attempts += 1
                if attempts > 500:
                    loc = True
                    print('Warning: Cannot find space for a new shape.')
                    return True
        return False


if __name__ == "__main__":
    pass
