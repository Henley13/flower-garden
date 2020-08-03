# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The flowers.dcgan subpackage includes functions to train and use a DCGAN.
"""

from .dcgan import build_generator_model
from .dcgan import build_discriminator_model
from .dcgan import train


_dcgan = [
    "build_generator_model",
    "build_discriminator_model",
    "train"]

__all__ = _dcgan
