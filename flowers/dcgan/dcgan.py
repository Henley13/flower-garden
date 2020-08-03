# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Deep Convolutional Generative Adversarial Networks.

Paper: "Unsupervised representation learning with Deep Convolutional
        Generative Adversarial Networks"
Authors: Radford, Alec
         Metz, Luke
         Chintala, Soumith
Year: 2015
Reference 1: https://arxiv.org/abs/1511.06434
Reference 2: https://www.tensorflow.org/tutorials/generative/dcgan
"""

import os
import time

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import (Dense, BatchNormalization, LeakyReLU,
                                     Conv2D, Reshape, Conv2DTranspose, Input,
                                     ReLU)

from skimage import img_as_ubyte

import flowers.utils as utils


# ### Generator ###

def build_generator_model(input_shape=(100,)):
    # define input layer
    input_ = Input(
        shape=input_shape,
        name="input_generator",
        dtype="float32")

    # weights initialization
    weight_init = RandomNormal(mean=0.0, stddev=0.02)
    gamma_init = RandomNormal(mean=1.0, stddev=0.02)

    # model
    shape0 = 4  # image_size / 16
    dense0 = Dense(
        units=shape0 * shape0 * 1024,
        use_bias=False,
        kernel_initializer=weight_init,
        name="dense0")(input_)
    dense0_bn = BatchNormalization(
        gamma_initializer=gamma_init,
        name="dense0_bn")(dense0)
    dense0_act = ReLU(
        name="dense0_act")(dense0_bn)
    deconv0_reshape = Reshape(
        target_shape=(shape0, shape0, 1024),
        name="dense0_reshape")(
        dense0_act)  # (?, image_size/16, image_size/16, 1024)

    # deconvolution 1
    deconv1 = Conv2DTranspose(
        filters=512,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=weight_init,
        name="deconv1")(deconv0_reshape)
    deconv1_bn = BatchNormalization(
        gamma_initializer=gamma_init,
        name="deconv1_bn")(deconv1)
    deconv1_act = ReLU(
        name="deconv1_act")(deconv1_bn)  # (?, image_size/8, image_size/8, 512)

    # deconvolution 2
    deconv2 = Conv2DTranspose(
        filters=256,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=weight_init,
        name="deconv2")(deconv1_act)
    deconv2_bn = BatchNormalization(
        gamma_initializer=gamma_init,
        name="deconv2_bn")(deconv2)
    deconv2_act = ReLU(
        name="deconv2_act")(deconv2_bn)  # (?, image_size/4, image_size/4, 256)

    # deconvolution 3
    deconv3 = Conv2DTranspose(
        filters=128,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=weight_init,
        name="deconv3")(deconv2_act)
    deconv3_bn = BatchNormalization(
        gamma_initializer=gamma_init,
        name="deconv3_bn")(deconv3)
    deconv3_act = ReLU(
        name="deconv3_act")(deconv3_bn)  # (?, image_size/2, image_size/2, 128)

    # deconvolution 4
    deconv4 = Conv2DTranspose(
        filters=3,
        kernel_size=(5, 5),
        strides=(2, 2),
        activation="tanh",
        padding="same",
        use_bias=False,
        kernel_initializer=weight_init,
        name="deconv4")(deconv3_act)  # (?, image_size, image_size, 3)

    # define generator model
    model = Model(
        inputs=input_,
        outputs=deconv4,
        name="generator")

    return model


# ### Discriminator ###

def build_discriminator_model(input_shape=(64, 64, 3)):
    # define input layer
    input_ = Input(
        shape=input_shape,
        name="input_discriminator",
        dtype="float32")

    # weights initialization
    weight_init = RandomNormal(mean=0.0, stddev=0.02)
    gamma_init = RandomNormal(mean=1.0, stddev=0.02)

    # model
    conv0 = Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer=weight_init,
        name="conv0")(input_)
    conv0_act = LeakyReLU(
        alpha=0.02,
        name="dense0_act")(conv0)  # (?, 32, 32, 128)

    conv1 = Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer=weight_init,
        name="conv1")(conv0_act)
    conv1_bn = BatchNormalization(
        gamma_initializer=gamma_init,
        name="conv1_bn")(conv1)
    conv1_act = LeakyReLU(
        alpha=0.02,
        name="conv1_act")(conv1_bn)  # (?, 16, 16, 256)

    conv2 = Conv2D(
        filters=512,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer=weight_init,
        name="conv2")(conv1_act)
    conv2_bn = BatchNormalization(
        gamma_initializer=gamma_init,
        name="conv2_bn")(conv2)
    conv2_act = LeakyReLU(
        alpha=0.02,
        name="conv2_act")(conv2_bn)  # (?, 8, 8, 512)

    conv3 = Conv2D(
        filters=1024,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer=weight_init,
        name="conv3")(conv2_act)
    conv3_bn = BatchNormalization(
        gamma_initializer=gamma_init,
        name="conv3_bn")(conv3)
    conv3_act = LeakyReLU(
        alpha=0.02,
        name="conv3_act")(conv3_bn)  # (?, 4, 4, 1024)

    conv4 = Conv2D(
        filters=1,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding='valid',
        activation="sigmoid",
        use_bias=False,
        kernel_initializer=weight_init,
        name="conv4")(conv3_act)  # (?, 1)

    # define discriminator model
    model = Model(
        inputs=input_,
        outputs=conv4,
        name="discriminator")

    return model


# ### Training functions ###


cross_entropy = BinaryCrossentropy(
    from_logits=True,
    name='binary_crossentropy')


def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return loss


def discriminator_loss(real_output, fake_output):
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    total_loss = real_loss + fake_loss

    return total_loss


generator_optimizer = Adam(
    learning_rate=0.0002,
    beta_1=0.5,
    name='Adam_G')
discriminator_optimizer = Adam(
    learning_rate=0.0002,
    beta_1=0.5,
    name='Adam_D')


@tf.function
def train_step(real_images, generator, discriminator,
               batch_size=128, noise_dim=100):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # compute gradients
    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    # apply backpropagation
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

    # compute metrics
    mean_gen_loss = Mean('loss_generator', dtype=tf.float32)(gen_loss)
    mean_disc_loss = Mean('loss_discriminator', dtype=tf.float32)(disc_loss)

    return mean_gen_loss, mean_disc_loss


def generate_and_plot(generator, input_noise,
                      path_output=None, ext="png", show=True):
    generated_images = generator(input_noise, training=False)
    generated_images = generated_images.numpy()
    generated_images = img_as_ubyte(generated_images)
    utils.plot_mosaic(generated_images, nb_row=5, nb_col=5, framesize=(10, 10),
                      path_output=path_output, ext=ext, show=show)

    return


def train(generator, discriminator, dataset, nb_epochs, training_directory):

    # initialize summaries
    tensorboard_directory = os.path.join(training_directory, "tensorboard")
    if not os.path.isdir(tensorboard_directory):
        os.mkdir(tensorboard_directory)
    summary_writer = tf.summary.create_file_writer(tensorboard_directory)

    # initialize checkpoints
    checkpoints_directory = os.path.join(training_directory, "checkpoints")
    if not os.path.isdir(checkpoints_directory):
        os.mkdir(checkpoints_directory)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)

    # loop over epochs
    mean_gen_loss, mean_disc_loss = None, None
    test_noise = tf.random.normal([25, 100])
    for epoch in range(nb_epochs):
        start = time.time()

        for batch_images in dataset:
            mean_gen_loss, mean_disc_loss = train_step(
                batch_images, generator, discriminator)

        # write summaries
        with summary_writer.as_default():
            tf.summary.scalar('loss_generator',
                              mean_gen_loss.result(),
                              step=epoch)
            tf.summary.scalar('loss_discriminator',
                              mean_disc_loss.result(),
                              step=epoch)

        # save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # plot a sample of generated images
        path = os.path.join(training_directory,
                            "generated_mosaic_{0}.png".format(epoch))
        generate_and_plot(generator, test_noise, path_output=path, show=False)

        # verbose
        end = time.time()
        duration = end - start
        print("Epoch {0} ({1} sec): Generator Loss {2} | "
              "Discriminator Loss {3}"
              .format(epoch, duration, mean_gen_loss, mean_disc_loss))

        # reset metrics at every epoch
        mean_gen_loss.reset_states()
        mean_disc_loss.reset_states()

    # save and plot at the end of the training
    checkpoint.save(file_prefix=checkpoint_prefix)
    path = os.path.join(training_directory, "generated_mosaic_final.png")
    generate_and_plot(generator, test_noise, path_output=path, show=False)

    return
