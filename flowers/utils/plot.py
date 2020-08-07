# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Function to plot images.
"""

import numpy as np
import matplotlib.pyplot as plt


# ### Mosaics ###

def plot_mosaic(images, nb_row, nb_col, framesize=(10, 10),
                path_output=None, ext="png", show=True):
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
    path_output : str or None
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    Returns
    -------

    """
    # initialize frame
    image_test = images[0]
    h_, w_, channel = image_test.shape
    h = h_ + 4
    w = w_ + 4
    #if channel != 3:
    #    raise ValueError("Images have {0} channels instead of 3."
    #                     .format(channel))
    if len(images) != nb_row * nb_col:
        raise ValueError("The mosaic requires {0}, but {1} are actually used."
                         .format(nb_row * nb_col, len(images)))
    frame = np.zeros((nb_row * h, nb_col * w, channel), dtype=np.uint8)

    # initialize plot
    plt.figure(figsize=framesize)

    # build frame
    i = 0
    for row in range(nb_row):
        for col in range(nb_col):
            image = images[i]
            if image.shape != (h_, w_, channel):
                raise ValueError("Image number {0} has a shape {1} instead "
                                 "of {2}"
                                 .format(i, image.shape, (h_, w_, channel)))
            image = np.pad(image, ((2, 2), (2, 2), (0, 0)))
            frame[row*h:(row+1)*h, col*w:(col+1)*w, :] = image
            i += 1
    frame = np.pad(frame, ((2, 2), (2, 2), (0, 0)))

    # plot
    plt.imshow(frame)

    # format plot
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # output
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


# ### Utilities functions ###

def save_plot(path_output, ext):
    """Save the plot.

    Parameters
    ----------
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # add extension at the end of the filename
    extension = "." + ext
    if extension not in path_output:
        path_output += extension

    # save the plot
    if isinstance(ext, str):
        # add extension at the end of the filename
        extension = "." + ext
        if extension not in path_output:
            path_output += extension
        plt.savefig(path_output, format=ext)
    elif isinstance(ext, list):
        for ext_ in ext:
            # add extension at the end of the filename
            extension = "." + ext_
            if extension not in path_output:
                path_output += extension
            plt.savefig(path_output, format=ext_)
    else:
        Warning("Plot is not saved because the extension is not valid: "
                "{0}.".format(ext))

    return
