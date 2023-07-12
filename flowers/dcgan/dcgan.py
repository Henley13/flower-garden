# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: MIT

"""
Deep Convolutional Generative Adversarial Networks.

Paper: "Unsupervised representation learning with Deep Convolutional
        Generative Adversarial Networks"
Authors: Radford, Alec
         Metz, Luke
         Chintala, Soumith
Year: 2015
Reference 1: https://arxiv.org/abs/1511.06434
Reference 2: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

import flowers.utils as utils


# ### Model ###

# custom weights initialization
def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# ### Generator ###


class Generator(nn.Module):
    def __init__(self, n_gpu, z_size, n_features, n_channel):
        super(Generator, self).__init__()
        # initialize parameters
        self.n_gpu = n_gpu
        self.z_size = z_size
        self.n_features = n_features
        self.n_channel = n_channel

        # build network
        self.main = nn.Sequential(
            # ``z_size``
            nn.ConvTranspose2d(
                self.z_size,
                self.n_features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(self.n_features * 8),
            nn.ReLU(True),
            # ``(n_features*8) x 4 x 4``
            nn.ConvTranspose2d(
                self.n_features * 8,
                self.n_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.n_features * 4),
            nn.ReLU(True),
            # ``(n_features*4) x 8 x 8``
            nn.ConvTranspose2d(
                self.n_features * 4,
                self.n_features * 2,
                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_features * 2),
            nn.ReLU(True),
            # ``(n_features*2) x 16 x 16``
            nn.ConvTranspose2d(
                self.n_features * 2,
                self.n_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(True),
            # ``n_features x 32 x 32``
            nn.ConvTranspose2d(
                self.n_features,
                self.n_channel,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.Tanh()
            # ``n_channel x 64 x 64``
        )

    def forward(self, z_input):
        return self.main(z_input)


# ### Discriminator ###


class Discriminator(nn.Module):
    def __init__(self, n_gpu, n_channel, n_features):
        super(Discriminator, self).__init__()
        # initialize parameters
        self.n_gpu = n_gpu
        self.n_channel = n_channel
        self.n_features = n_features

        # build network
        self.main = nn.Sequential(
            # ``n_channel x 64 x 64``
            nn.Conv2d(
                self.n_channel,
                self.n_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ``n_features x 32 x 32``
            nn.Conv2d(
                self.n_features,
                self.n_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.n_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ``(n_features*2) x 16 x 16``
            nn.Conv2d(
                self.n_features * 2,
                self.n_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.n_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ``(n_features*4) x 8 x 8``
            nn.Conv2d(
                self.n_features * 4,
                self.n_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.n_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ``(n_features*8) x 4 x 4``
            nn.Conv2d(
                self.n_features * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_image):
        return self.main(input_image)
