# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The flowers.dcgan subpackage includes functions to train and use a DCGAN.
"""

from .dcgan import init_weights
from .dcgan import Generator
from .dcgan import Discriminator


_dcgan = [
    "init_weights",
    "Generator",
    "Discriminator"
]

__all__ = _dcgan
