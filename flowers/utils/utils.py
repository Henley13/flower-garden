# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""Routine functions."""

import os
import time
import datetime
import sys
import inspect
import shutil


# ### Sanity checks functions ###

def check_parameter(**kwargs):
    """Check dtype of the function's parameters.

    Parameters
    ----------
    kwargs : Type or Tuple[Type]
        Map of each parameter with its expected dtype.

    Returns
    -------
    _ : bool
        Assert if the array is well formatted.

    """
    # get the frame and the parameters of the function
    frame = inspect.currentframe().f_back
    _, _, _, values = inspect.getargvalues(frame)

    # compare each parameter with its expected dtype
    for arg in kwargs:
        expected_dtype = kwargs[arg]
        parameter = values[arg]
        if not isinstance(parameter, expected_dtype):
            actual = "'{0}'".format(type(parameter).__name__)
            if isinstance(expected_dtype, tuple):
                target = ["'{0}'".format(x.__name__) for x in expected_dtype]
                target = "(" + ", ".join(target) + ")"
            else:
                target = expected_dtype.__name__
            raise TypeError("Parameter {0} should be a {1}. It is a {2} "
                            "instead.".format(arg, target, actual))

    return True


def check_directories(path_directories):
    # check directories exist
    check_parameter(path_directories=list)
    for path_directory in path_directories:
        if not os.path.isdir(path_directory):
            raise ValueError("Directory does not exist: {0}"
                             .format(path_directory))

    return


# ### Script routines ###

def initialize_script(log_directory, experiment_name=None):
    # check parameters
    check_parameter(log_directory=str)
    check_parameter(experiment_name=(str, type(None)))

    # get filename of the script that call this function
    try:
        previous_filename = inspect.getframeinfo(sys._getframe(1))[0]
    except ValueError:
        previous_filename = None

    # get date of execution
    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d-%H%M%S")

    # format log name
    if experiment_name is not None:
        log_name = "{0}-{1}".format(date, experiment_name)
    else:
        log_name = "{0}-script".format(date)

    # initialize logging in a specific log directory
    path_log_directory = os.path.join(log_directory, log_name)
    os.mkdir(path_log_directory)
    path_log_file = os.path.join(path_log_directory, "log")
    sys.stdout = Logger(path_log_file)

    # copy python script in the log directory
    if previous_filename is not None:
        path_output = os.path.join(path_log_directory,
                                   os.path.basename(previous_filename))
        shutil.copyfile(previous_filename, path_output)

    # print information about launched script
    if previous_filename is not None:
        print("Running {0} file..."
              .format(os.path.basename(previous_filename)))
        print()
    start_time = time.time()
    if experiment_name is not None:
        print("Experiment name: {0}".format(experiment_name))
    print("Log directory: {0}".format(log_directory))
    print("Log name: {0}".format(log_name))
    print("Date: {0}".format(date), "\n")

    return start_time, path_log_directory


def end_script(start_time):
    # check parameters
    check_parameter(start_time=(int, float))

    # time the script
    end_time = time.time()
    duration = int(round((end_time - start_time) / 60))
    print("Duration: {0} minutes".format(duration), "\n")

    return


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
