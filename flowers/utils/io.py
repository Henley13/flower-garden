# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Function used to read data from various sources and store them in a
multidimensional tensor (np.ndarray).
"""

import os
import warnings
import hashlib

import numpy as np
from skimage import io

from torch.utils.data import Dataset
from torchvision.io import read_image as torch_read_image


# ### Dataset ###


class CustomImageDataset(Dataset):
    def __init__(self,
                 metadata_file,
                 image_directory,
                 transform=None,
                 target_transform=None):
        self.image_metadata = metadata_file
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_metadata)

    def __getitem__(self, idx):
        # get image and label
        filename = self.image_metadata.loc[idx, "filename"]
        path_image = os.path.join(self.image_directory, filename)
        image = read_image(path_image)
        label = int(self.image_metadata.loc[idx, "label_int"])

        # transform if necessary
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


# ### Read ###


def read_image(path):
    """Read an image with the .png, .jpg or .jpeg extension.

    The input image should be in 2-d with unsigned integer 8 or 16
    bits, integer

    Parameters
    ----------
    path : str
        Path of the image to read.

    Returns
    -------
    tensor : ndarray, np.uint or np.int
        A 2-d or 3-d tensor with spatial dimensions.

    """
    # read image
    tensor = io.imread(path)

    return tensor


def parse_images():

    yield


# ### Write ###

def save_image(image, path):
    """Save a 2-d image.

    Parameters
    ----------
    image : np.ndarray
        Tensor to save with shape (z, y, x) or (y, x).
    path : str
        Path of the saved image.

    Returns
    -------

    """
    # save image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(path, image)

    return


# ### Hash ###

def get_md5_number(a):
    """Get a number from the md5 hash value expressed in hexadecimal.

    Parameters
    ----------
    a : python object
        Object used to compute the md5 value.

    Returns
    -------
    x : int
        md5 value.
    """
    # get md5
    md5 = hashlib.md5(a)
    x = int(md5.hexdigest(), 16)

    return x
