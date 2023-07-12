# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The flowers.utils subpackage includes functions to load data, plot and several
utilities functions.
"""

from .io import CustomImageDataset
from .io import read_image
from .io import save_image
from .io import get_md5_number
from .io import parse_images

from .plot import plot_mosaic

from .utils import check_parameter
from .utils import check_directories
from .utils import initialize_script
from .utils import end_script


_io = [
    "CustomImageDataset",
    "read_image",
    "save_image",
    "get_md5_number",
    "parse_images"]

_plot = [
    "plot_mosaic"]

_utils = [
    "check_parameter",
    "check_directories",
    "initialize_script",
    "end_script"]

__all__ = _io + _plot + _utils
