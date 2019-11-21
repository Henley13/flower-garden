# -*- coding: utf-8 -*-

"""
Function to plot images.
"""

import numpy as np
import matplotlib.pyplot as plt


# ### Read ###

def plot_mosaic(images, nb_row, nb_col, framesize=(10, 10)):
    """Plot a mosaic of images.

    Parameters
    ----------
    images : np.ndarray or list
        Array-like with images of the same shape (height, width, channel). If
        'images' is an array, the first dimension should index the images.
    nb_row : int
        Number of rows in the mosaic.
    nb_col : int
        Number of columns in the mosaic.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.

    Returns
    -------

    """
    # initialize frame
    image_test = images[0]
    h, w, channel = image_test.shape
    if channel != 3:
        raise ValueError("Images have {0} channels instead of 3."
                         .format(channel))
    if len(images) != nb_row * nb_col:
        raise ValueError("The mosaic requires {0}, but {1} are actually used."
                         .format(nb_row * nb_col, len(images)))
    frame = np.zeros((nb_row * h, nb_col * w, 3), dtype=np.uint8)

    # initialize plot
    plt.figure(figsize=framesize)

    # build frame
    i = 0
    for row in range(nb_row):
        for col in range(nb_col):
            image = images[i]
            if image.shape != (h, w, 3):
                raise ValueError("Image number {0} has a shape {1} instead "
                                 "of {2}".format(i, image.shape, (h, w, 3)))
            frame[row*h:(row+1)*h, col*w:(col+1)*w, :] = image
            i += 1

    # plot
    plt.imshow(frame)

    # format plot
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.show()

    return
