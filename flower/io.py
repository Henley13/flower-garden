# -*- coding: utf-8 -*-

"""
Function used to read data from various sources and store them in a
multidimensional tensor (np.ndarray).
"""

import warnings
import numpy as np
from skimage import io


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
    # TODO allow more input dtype
    # read image
    tensor = io.imread(path)

    return tensor


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
