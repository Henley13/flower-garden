# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""Script to run to train a DCGAN model."""

import os
import argparse
import tensorflow as tf

import pandas as pd
import numpy as np

import flowers.utils as utils

from dcgan import build_generator_model, build_discriminator_model, train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    print()
    print("Run script train.py")
    print("TensorFlow version: {0}".format(tf.__version__))

    # get GPU devices
    gpu_devices = tf.config.experimental.list_logical_devices('GPU')
    print("Number of GPUs: ", len(gpu_devices))
    for device in gpu_devices:
        print("\r", device.name)
    print()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory",
                        help="Path of the input directory.",
                        type=str)
    parser.add_argument("output_directory",
                        help="Path of the output directory.",
                        type=str)
    parser.add_argument("--log_directory",
                        help="Path of the log directory.",
                        type=str,
                        default="/Users/arthur/output/flowers/log")

    # initialize parameters
    args = parser.parse_args()
    input_directory = args.input_directory
    output_directory = args.output_directory
    log_directory = args.log_directory
    nb_epochs = 50
    batch_size = 128
    noise_dim = 100

    # check directories exists
    utils.check_directories([input_directory, output_directory, log_directory])

    # initialize script
    start_time, training_directory = utils.initialize_script(log_directory)

    # prepare dataset
    # print("Prepare dataset...")
    # path = os.path.join(input_directory, "data.csv")
    # df = pd.read_csv(path, sep=";", encoding="utf-8")
    # filenames = list(df.loc[:, "filename"])
    # images = np.zeros((len(filenames), 64, 64, 3), dtype=np.uint8)
    # for i, filename in enumerate(filenames):
    #     path = os.path.join(input_directory, "data", filename)
    #     image = utils.read_image(path)
    #     image = tf.image.resize(image, (64, 64), antialias=True)
    #     images[i] = image
    # X_train = images.copy()
    # X_train = X_train.astype(np.float32)
    # X_train = (X_train - 127.5) / 127.5
    # dataset = tf.data.Dataset.from_tensor_slices(X_train)
    # dataset = dataset.shuffle(buffer_size=60000, seed=13)
    # dataset = dataset.batch(batch_size=batch_size)
    # print("\r Dataset length: {0}".format(len(X_train)), "\n")

    print("Prepare dataset...")
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = tf.image.resize(X_train, (64, 64), antialias=True)
    X_train = X_train.astype('float32')
    X_train = (X_train - 127.5) / 127.5
    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(buffer_size=60000, seed=13)
    dataset = dataset.batch(batch_size=batch_size)
    print("\r Dataset length: {0}".format(len(X_train)), "\n")

    # build models
    print("Build model...", "\n")
    generator = build_generator_model()
    discriminator = build_discriminator_model()

    # train model
    print("Train model...", "\n")
    train(generator, discriminator, dataset,
          nb_epochs, batch_size, noise_dim,
          training_directory)

    print()
    utils.end_script(start_time)
