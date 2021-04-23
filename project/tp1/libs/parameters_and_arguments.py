#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Deep Learning - Assignment #1:
- Pokemon Image Classification and Segmentation with Deep Neural Networks

Integrated Master of Computer Science and Engineering

NOVA School of Science and Technology,
NOVA University of Lisbon - 2020/2021

Authors:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt) - Student no. 49067
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt) - Student no. 42648

Instructor(s):
- Ludwig Krippahl (ludi@fct.unl.pt)
- Claudia Soares (claudia.soares@fct.unl.pt)

Parameters' and Arguments' Module for the Project

"""

# Import Python's Modules, Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Load Txt Function from the NumPy Python's Library
from numpy import loadtxt

# Import the Multi-Processing Python's Module as multiprocessing alias
import multiprocessing as multiprocessing

# Import the Tensorflow as Tensorflow alias Python's Module
import tensorflow as tensorflow


# Constants

# The boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs)
TENSORFLOW_KERAS_HPC_BACKEND_SESSION = True

# The Number of CPU's Processors/Cores
NUM_CPU_PROCESSORS_CORES = multiprocessing.cpu_count()

# The Number of GPU's Devices
NUM_GPU_DEVICES = len(tensorflow.config.list_physical_devices('GPU'))

# The Number of Examples (Images/Masks) for the final Training Set
NUM_EXAMPLES_FINAL_TRAINING_SET = 3500

# The Number of Examples (Images/Masks) for the final Validation Set
NUM_EXAMPLES_FINAL_VALIDATION_SET = 500

# The Number of Examples (Images/Masks) for the final Testing Set
NUM_EXAMPLES_FINAL_TESTING_SET = 500

# The Height of the Images/Pictures (64 pixels)
IMAGES_HEIGHT = 64

# The Width of the Images/Pictures (64 pixels)
IMAGES_WIDTH = 64

# The Size of the Images/Pictures (64 x 64 pixels)
IMAGE_SIZE = (IMAGES_WIDTH * IMAGES_HEIGHT)

# The Number of Channels for RGB Colors
NUM_CHANNELS_RGB = 3

# The Number of Channels for Grayscale Colors
NUM_CHANNELS_GRAY_SCALE = 1

# The Number of Filters for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
NUM_FILTERS_PER_BLOCK = [32, 64, 128, 256]

# The Height of the Kernel of the Filters used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
KERNEL_HEIGHT = 3

# The Width of the Kernel of the Filters used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
KERNEL_WIDTH = 3

# The Height of the Pooling Matrix used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
POOLING_HEIGHT = 2

# The Width of the Pooling Matrix used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
POOLING_WIDTH = 2

# The Height of the Stride used on
# the Pooling Matrices used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
STRIDE_HEIGHT = 2

# The Width of the Stride used on
# the Pooling Matrices used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
STRIDE_WIDTH = 2

# The Number of Units of the last Dens Layer for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
NUM_UNITS_LAST_DENSE_LAYER = 512

# The Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
AVAILABLE_OPTIMISERS_LIST = ['SGD', 'RMSPROP', 'ADAM', 'ADAGRAD', 'ADADELTA', 'ADAMAX']

# The Number of Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
NUM_AVAILABLE_OPTIMISERS = len(AVAILABLE_OPTIMISERS_LIST)

# The Learning Rates for the Optimisers used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
INITIAL_LEARNING_RATES = [0.005, 0.0005, 0.00041, 0.012, 0.25, 0.001]

# The Matplotlib Colors for the Optimisers used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
OPTIMISERS_COLORS_MATPLOTLIB = ['red', 'darkorange', 'forestgreen', 'midnightblue', 'magenta', 'black']

# The Momentum #1 for the Optimisers used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
MOMENTUM_1 = 0.9

# The Momentum #2 for the Optimiser used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
MOMENTUM_2 = 0.0

# The Number of Epochs for the Optimiser for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
NUM_EPOCHS = 50

# The Number of Last Epochs to be discarded for the Early Stopping for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
NUM_LAST_EPOCHS_TO_BE_DISCARDED_FOR_EARLY_STOPPING = 0

# The Size of the Batch for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
BATCH_SIZE = 16

# The Number of Classes for the Datasets from the Pokemons' Data
NUM_CLASSES_POKEMON_TYPES = len(loadtxt('./dataset/pokemon_types.txt', dtype='str'))

# The Models available to use for the
# feed-forward Convolution Neural Network (C.N.N.)
AVAILABLE_MODELS_LIST = ['MODEL0', 'MODEL1']

# The Number of Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
NUM_AVAILABLE_MODELS = len(AVAILABLE_MODELS_LIST)