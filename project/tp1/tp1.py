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

Main Module for the Project

"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Python's Modules, Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Load Txt Function from the NumPy Python's Library
from numpy import loadtxt

# Import Mean (Average) Function from the NumPy Python's Library
from numpy import mean

# Import Date Time from Date Time Python's System Module
from datetime import datetime as date_time

# Import the Multi-Processing Python's Module as multiprocessing alias
import multiprocessing as multiprocessing

# Import PyPlot from the Matplotlib Python's Library
from matplotlib import pyplot as py_plot

# Import the Tensorflow as Tensorflow alias Python's Module
import tensorflow as tensorflow

# Import the Backend Module from the TensorFlow.Python.Keras Python's Module
from tensorflow.python.keras import backend as keras_backend

# Import the Image's Data Generator from
# the Tensorflow.Keras.PreProcessing.Image Python's Module
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import the Sequential from the TensorFlow.Keras.Models Python's Module
from tensorflow.keras.models import Sequential

# Import the Convolution 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Conv2D

# Import the Activation Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Activation

# Import the Max Pooling 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import MaxPooling2D

# Import the Flatten Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Flatten

# Import the Dense Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Dense

# Import the Dropout Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Dropout

# Import the Stochastic Gradient Descent (S.G.D.) Optimiser
# from the TensorFlow.Keras.Optimisers Python's Module
from tensorflow.keras.optimizers import SGD

# Import the Root Mean Squared Prop (R.M.S. PROP) Optimiser
# from the TensorFlow.Keras.Optimisers Python's Module
from tensorflow.keras.optimizers import RMSprop

# Import the ADAptive Moment estimation (ADA.M.) Optimiser
# from the TensorFlow.Keras.Optimisers Python's Module
from tensorflow.keras.optimizers import Adam

# Import the ADAptive GRADient algorithm (ADA.GRAD.) Optimiser
# from the TensorFlow.Keras.Optimisers Python's Module
from tensorflow.keras.optimizers import Adagrad

# Import the ADAptive DELTA algorithm (ADA.DELTA) Optimiser
# from the TensorFlow.Keras.Optimisers Python's Module
from tensorflow.keras.optimizers import Adadelta

# Import the ADAptive MAX algorithm (ADA.MAX.) Optimiser
# from the TensorFlow.Keras.Optimisers Python's Module
from tensorflow.keras.optimizers import Adamax

# Import the Early Stopping from
# the TensorFlow.Keras.Optimisers Python's Module
from tensorflow.keras.callbacks import EarlyStopping

# Import the TensorBoard from
# the TensorFlow.Keras.Callbacks Python's Module
from tensorflow.keras.callbacks import TensorBoard

# Import the Categorical Cross-Entropy from
# the TensorFlow.Keras.Metrics Python's Module
from tensorflow.keras.metrics import categorical_crossentropy

# Import the Categorical Accuracy from
# the TensorFlow.Keras.Metrics Python's Module
from tensorflow.keras.metrics import categorical_accuracy

# Import the Auxiliary Function to Load the Data for the Project,
# from the TP1_Utils' Python's Module
from tp1_utils import load_data

# Constants

# The boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs)
TENSORFLOW_KERAS_HPC_BACKEND_SESSION = True

# The number of CPU's Processors/Cores
NUM_CPU_PROCESSORS_CORES = multiprocessing.cpu_count()

# The number of GPU's Devices
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
AVAILABLE_OPTIMISERS_LIST = ["SGD", "RMSPROP", "ADAM", "ADAGRAD", "ADADELTA", "ADAMAX"]

# The Number of Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
NUM_AVAILABLE_OPTIMISERS = len(AVAILABLE_OPTIMISERS_LIST)

# The Learning Rates for the Optimisers used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
INITIAL_LEARNING_RATES = [0.005, 0.0005, 0.00041, 0.012, 0.25, 0.001]

# The Matplotlib Colors for the Optimisers used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
OPTIMISERS_COLORS_MATPLOTLIB = ["red", "darkorange", "forestgreen", "midnightblue", "magenta", "black"]

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
NUM_CLASSES_POKEMON_TYPES = len(loadtxt("./dataset/pokemon_types.txt", dtype='str'))


# Retrieve the Datasets from the Pokemons' Data,
# in order to be used to build the model for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
def retrieve_datasets_from_pokemon_data():

    # Load the Data needed for the Project, using the given Auxiliary Function for it,
    # containing all the Datasets related to the Pokemons' Data
    pokemon_datasets = load_data()

    # Retrieve the xs (features) of the Pokemons' Datasets, for the Training Set
    xs_features_training_set = pokemon_datasets['train_X'][:NUM_EXAMPLES_FINAL_TRAINING_SET]

    # Retrieve the xs (masks) of the Pokemons' Datasets, for the Training Set
    xs_masks_training_set = pokemon_datasets['train_masks'][:NUM_EXAMPLES_FINAL_TRAINING_SET]

    # Retrieve the ys (classes) of the Pokemons' Datasets, for the Training Set
    ys_classes_training_set = pokemon_datasets['train_classes'][:NUM_EXAMPLES_FINAL_TRAINING_SET]

    # Retrieve the ys (labels) of the Pokemons' Datasets, for the Training Set
    ys_labels_training_set = pokemon_datasets['train_labels'][:NUM_EXAMPLES_FINAL_TRAINING_SET]

    # Retrieve the xs (features) of the Pokemons' Datasets, for the Validation Set
    xs_features_validation_set = pokemon_datasets['train_X'][NUM_EXAMPLES_FINAL_TRAINING_SET:]

    # Retrieve the xs (masks) of the Pokemons' Datasets, for the Validation Set
    xs_masks_validation_set = pokemon_datasets['train_masks'][NUM_EXAMPLES_FINAL_TRAINING_SET:]

    # Retrieve the ys (classes) of the Pokemons' Datasets, for the Validation Set
    ys_classes_validation_set = pokemon_datasets['train_classes'][NUM_EXAMPLES_FINAL_TRAINING_SET:]

    # Retrieve the ys (labels) of the Pokemons' Datasets, for the Validation Set
    ys_labels_validation_set = pokemon_datasets['train_labels'][NUM_EXAMPLES_FINAL_TRAINING_SET:]

    # Retrieve the xs (features) of the Pokemons' Datasets, for the Testing Set
    xs_features_testing_set = pokemon_datasets['test_X']

    # Retrieve the xs (masks) of the Pokemons' Datasets, for the Testing Set
    xs_masks_testing_set = pokemon_datasets['test_masks']

    # Retrieve the ys (classes) of the Pokemons' Datasets, for the Testing Set
    ys_classes_testing_set = pokemon_datasets['test_classes']

    # Retrieve the ys (labels) of the Pokemons' Datasets, for the Testing Set
    ys_labels_testing_set = pokemon_datasets['test_labels']

    # Return all the retrieved Datasets related to the Pokemons' Data
    return xs_features_training_set, xs_masks_training_set, ys_classes_training_set, ys_labels_training_set, \
        xs_features_validation_set, xs_masks_validation_set, ys_classes_validation_set, ys_labels_validation_set, \
        xs_features_testing_set, xs_masks_testing_set, ys_classes_testing_set, ys_labels_testing_set


# Function to create the Images' Data Generator for Pre-Processing with Data Augmentation
def image_data_generator_for_preprocessing_with_data_augmentation():
    # Create the Images' Data Generator for Pre-Processing with Data Augmentation
    image_data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    # Return the Images' Data Generator for Pre-Processing with Data Augmentation
    return image_data_generator


# Function to create the need Early Stopping Callbacks for
# the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification
def create_early_stopping_callbacks():
    # Create the Callback for Early Stopping, related to
    # the Loss Cost of the Fitting/Training with Training Set
    training_loss_early_stopping_callback = \
        EarlyStopping(
            monitor="loss",
            min_delta=1e-6,
            patience=(NUM_EPOCHS - NUM_LAST_EPOCHS_TO_BE_DISCARDED_FOR_EARLY_STOPPING),
            verbose=1,
            mode="min",
            baseline=0.08,
            restore_best_weights=True
        )

    # Create the Callback for Early Stopping, related to
    # the Accuracy of the Fitting/Training with Training Set
    training_accuracy_early_stopping_callback = \
        EarlyStopping(
            monitor="loss",
            min_delta=1e-6,
            patience=(NUM_EPOCHS - NUM_LAST_EPOCHS_TO_BE_DISCARDED_FOR_EARLY_STOPPING),
            verbose=1,
            mode="min",
            baseline=0.96,
            restore_best_weights=True
        )

    # Create the Callback for Early Stopping, related to
    # the Loss Cost of the Fitting/Training with Validation Set
    validation_loss_early_stopping_callback = \
        EarlyStopping(
            monitor="loss",
            min_delta=1e-6,
            patience=(NUM_EPOCHS - NUM_LAST_EPOCHS_TO_BE_DISCARDED_FOR_EARLY_STOPPING),
            verbose=1,
            mode="min",
            baseline=0.08,
            restore_best_weights=True
        )

    # Create the Callback for Early Stopping, related to
    # the Accuracy of the Fitting/Training with Validation Set
    validation_accuracy_early_stopping_callback = \
        EarlyStopping(
            monitor="loss",
            min_delta=1e-6,
            patience=(NUM_EPOCHS - NUM_LAST_EPOCHS_TO_BE_DISCARDED_FOR_EARLY_STOPPING),
            verbose=1,
            mode="min",
            baseline=0.96,
            restore_best_weights=True
        )

    # Return need Early Stopping Callbacks for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    return training_loss_early_stopping_callback, \
        training_accuracy_early_stopping_callback, \
        validation_loss_early_stopping_callback, \
        validation_accuracy_early_stopping_callback


# Function to create the Optimiser to be used for
# the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification
def create_optimiser(optimiser_id):

    # Initialise the Optimiser to be used for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    optimiser = None

    # It is being used the Stochastic Gradient Descent (S.G.D.) Optimiser
    if optimiser_id == AVAILABLE_OPTIMISERS_LIST[0]:

        # Initialise the Stochastic Gradient Descent (S.G.D.) Optimiser,
        # with the Learning Rate of 0.5% and Momentum of 90%
        optimiser = SGD(learning_rate=INITIAL_LEARNING_RATES[0],
                        momentum=MOMENTUM_1, decay=(INITIAL_LEARNING_RATES[0] / NUM_EPOCHS))

    # It is being used the Root Mean Squared Prop (R.M.S. PROP) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[1]:

        # Initialise the Root Mean Squared Prop (R.M.S. PROP) Optimiser,
        # with the Learning Rate of 0.5% and Momentum of 90%
        optimiser = RMSprop(learning_rate=INITIAL_LEARNING_RATES[1], momentum=MOMENTUM_2)

    # It is being used the ADAptive Moment estimation (ADA.M.) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[2]:

        # Initialise the ADAptive Moment estimation (ADA.M.) Optimiser,
        # with the Learning Rate of 0.5%
        optimiser = Adam(learning_rate=INITIAL_LEARNING_RATES[2], decay=0.1)

    # It is being used the ADAptive GRADient algorithm (ADA.GRAD.) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[3]:

        # Initialise the ADAptive GRADient algorithm (ADA.GRAD.) Optimiser,
        # with the Learning Rate of 0.5%
        optimiser = Adagrad(learning_rate=INITIAL_LEARNING_RATES[3])

    # It is being used the ADAptive DELTA algorithm (ADA.DELTA) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[4]:

        # Initialise the ADAptive DELTA algorithm (ADA.DELTA) Optimiser,
        # with the Learning Rate of 0.5%
        optimiser = Adadelta(learning_rate=INITIAL_LEARNING_RATES[4])

    # It is being used the ADAptive DELTA algorithm (ADA.DELTA) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[5]:

        # Initialise the ADAptive MAX algorithm (ADA.MAX.) Optimiser,
        # with the Learning Rate of 0.5%
        optimiser = Adamax(learning_rate=INITIAL_LEARNING_RATES[5])

    # Return the Optimiser to be used for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    return optimiser


# Function to create a Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification
def create_cnn_model_in_keras_sequential_api_for_image_classification(optimiser_id):

    # Create a Model for a feed-forward Convolution Neural Network (C.N.N.),
    # which is most appropriate for this type of problem (i.e., Image Classification),
    # using the Tensorflow Keras' Sequential API
    convolution_neural_network_tensorflow_keras_sequential_model = \
        Sequential(name="pokemon-images-multi-classes-classification")

    # --- 1st Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 1st Convolution 2D Layer, for the Input features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), using 32 Filters of a Kernel 3x3,
    # Same Padding and an Input Shape of (64 x 64 pixels), as also,
    # 3 Input Dimensions (for each Color Channel - RGB Color)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(NUM_FILTERS_PER_BLOCK[0], (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # --- 2nd Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 2nd Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 64 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(NUM_FILTERS_PER_BLOCK[1], (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a 3rd Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 64 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(NUM_FILTERS_PER_BLOCK[1], (STRIDE_HEIGHT, STRIDE_WIDTH),
                    padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # --- 3rd Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 4th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(NUM_FILTERS_PER_BLOCK[2], (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a 5th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(NUM_FILTERS_PER_BLOCK[2], (KERNEL_HEIGHT, KERNEL_WIDTH), padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # --- 4th Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 6th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(NUM_FILTERS_PER_BLOCK[3], (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a 7th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(NUM_FILTERS_PER_BLOCK[3], (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a 8th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(NUM_FILTERS_PER_BLOCK[3], (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # --- 5th Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.),
    # for the Softmax Classifier, for the Multi-Class Problem ---

    # Add a Flatten Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.), with 512 Units
    # NOTE:
    # - This is needed to flatten the input into a single dimension for the features,
    #   which is what the next Dense Layer needs;
    convolution_neural_network_tensorflow_keras_sequential_model.add(Flatten())

    # Add a Dense Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # for a total of 512 Units (Weights and Biases)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Dense(NUM_UNITS_LAST_DENSE_LAYER))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation("relu"))

    # Add a Dense Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # for a total of 512 Units (Weights and Biases)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Dense(NUM_UNITS_LAST_DENSE_LAYER))

    # It is being used the ADAptive GRADient algorithm (ADA.GRAD.) Optimiser
    if optimiser_id == AVAILABLE_OPTIMISERS_LIST[3]:

        # Add a Dropout Layer of 50%, for the Regularization of Hyper-Parameters,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        convolution_neural_network_tensorflow_keras_sequential_model.add(Dropout(0.5))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation("relu"))

    # Add a Dense Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # for a total of 10 Units (Weights and Biases), for each type of Pokemon
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Dense(NUM_CLASSES_POKEMON_TYPES))

    # Add a Softmax as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), for the Multi-Class Classifier
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation("softmax"))

    # Return the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    return convolution_neural_network_tensorflow_keras_sequential_model


# Function to plot the Training's and Validation's Losses,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
def plot_training_and_validation_losses_multi_classes_problem(cnn_model_in_keras_sequential_api_training_history,
                                                              optimiser_id, now_date_time_format,
                                                              plotting_style="seaborn-dark",
                                                              is_to_show=False):
    # Retrieve the Loss Values, from the Fitting/Training of
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem,
    # regarding the Training Set
    training_accuracy_values = \
        cnn_model_in_keras_sequential_api_training_history.history["loss"]

    # Retrieve the Loss Values, from the Fitting/Training of
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem,
    # regarding the Validation Set
    validation_accuracy_values = \
        cnn_model_in_keras_sequential_api_training_history.history["val_loss"]

    # Set the Style of the Plots, as 'Seaborn Dark' Style, by default
    py_plot.style.use(plotting_style)

    # Initialise the Plot Frame
    py_plot.figure(figsize=(8, 8), frameon=True)

    # Plot the Loss Values for the Training Set,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.plot(training_accuracy_values, "-", color="blue")

    # Plot the Loss Values for the Validation Set,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.plot(validation_accuracy_values, "-", color="red")

    # Plot the Title for the comparison of
    # the Loss Values for the Training and Validation Sets,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.title("Losses of the Model for a feed-forward Convolution Neural Network (C.N.N.)\n"
                  "for the Pokemon Images/Data Classification, for the Multi-Class Problem\n"
                  "with {} Optimiser".format(optimiser_id))

    # Set the X-Axis of the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.xlim((0, NUM_EPOCHS))

    # Plot the title for the X-Axis for the Number of Epochs,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.xlabel("Epoch no.")

    # Plot the title for the Y-Axis for the Loss Values',
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.ylabel("Loss Value")

    # Plot the legend for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.legend(["Training", "Validation"], loc="upper right", frameon=True)

    # Add a Grid to the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.grid(color="white", linestyle="--", linewidth=0.8)

    # Save the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.savefig("./files/images/plots/multi-classes-classification/loss/"
                    "loss-values-plot-{}-optimiser-{}.png".format(optimiser_id.lower(), now_date_time_format))

    # If it is supposed to show the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    if is_to_show:

        # Show the final Plot for the Loss Values' Comparison,
        # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
        # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
        py_plot.show()

    # Close the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.close()


# Function to plot the Training's and Validation's Accuracies,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
def plot_training_and_validation_accuracies_multi_classes_problem(cnn_model_in_keras_sequential_api_training_history,
                                                                  optimiser_id, now_date_time_format,
                                                                  plotting_style="seaborn-dark",
                                                                  is_to_show=False):
    # Retrieve the Accuracy Values, from the Fitting/Training of
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem,
    # regarding the Training Set
    training_accuracy_values = \
        cnn_model_in_keras_sequential_api_training_history.history["accuracy"]

    # Retrieve the Accuracy Values, from the Fitting/Training of
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem,
    # regarding the Validation Set
    validation_accuracy_values = \
        cnn_model_in_keras_sequential_api_training_history.history["val_accuracy"]

    # Set the Style of the Plots, as 'Seaborn Dark' Style, by default
    py_plot.style.use(plotting_style)

    # Initialise the Plot Frame
    py_plot.figure(figsize=(8, 8), frameon=True)

    # Plot the Accuracy Values for the Training Set,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.plot(training_accuracy_values, "-", color="blue")

    # Plot the Accuracy Values for the Validation Set,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.plot(validation_accuracy_values, "-", color="red")

    # Plot the Title for the comparison of
    # the Accuracy Values for the Training and Validation Sets,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.title("Accuracies of the Model for a feed-forward Convolution Neural Network (C.N.N.)\n"
                  "for the Pokemon Images/Data Classification, for the Multi-Class Problem\n"
                  "with {} Optimiser".format(optimiser_id))

    # Set the X-Axis of the final Plot for the Accuracy Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.xlim((0, NUM_EPOCHS))

    # Plot the title for the X-Axis for the Number of Epochs,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.xlabel("Epoch no.")

    # Plot the title for the Y-Axis for the Accuracy Values',
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.ylabel("Accuracy Value")

    # Plot the legend for the Accuracy Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.legend(["Training", "Validation"], loc="upper right", frameon=True)

    # Add a Grid to the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.grid(color="white", linestyle="--", linewidth=0.8)

    # Save the final Plot for the Accuracy Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.savefig("./files/images/plots/multi-classes-classification/accuracy/"
                    "accuracy-values-plot-{}-optimiser-{}.png".format(optimiser_id.lower(), now_date_time_format))

    # If it is supposed to show the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    if is_to_show:
        # Show the final Plot for the Loss Values' Comparison,
        # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
        # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
        py_plot.show()

    # Close the final Plot for the Accuracy Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.close()


# Function to plot the Training's and Validation's Losses,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
def plot_subset_metric_multi_classes_problem_all_optimisers(
        cnn_model_in_keras_sequential_api_optimisers_metric_history, subset, metric,
        now_date_time_format, plotting_style="seaborn-dark", is_to_show=False):

    # Set the Style of the Plots, as 'Seaborn Dark' Style, by default
    py_plot.style.use(plotting_style)

    # Initialise the Plot Frame
    py_plot.figure(figsize=(8, 8), frameon=True)

    # Initialise the list of legends for the Plots
    legends = []

    # For each available Optimiser
    for num_current_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

        # Append the Legend for the the current available Optimiser
        legends.append("{} [ lr = {} ]".format(AVAILABLE_OPTIMISERS_LIST[num_current_optimiser],
                                               INITIAL_LEARNING_RATES[num_current_optimiser]))

        # Plot the given Metrics' Values for the Subset given,
        # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
        # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
        py_plot.plot(cnn_model_in_keras_sequential_api_optimisers_metric_history[num_current_optimiser],
                     "-", color=OPTIMISERS_COLORS_MATPLOTLIB[num_current_optimiser])

    # Plot the Title for the comparison of
    # the Metric Values for the given Subset,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.title("Comparison of the {} {} of the Model for a feed-forward\nConvolution Neural Network (C.N.N.) "
                  "for the Pokemon Images/Data Classification,\n for the Multi-Class Problem "
                  "with all Optimisers".format(subset, metric))

    # Set the X-Axis of the final Plot for the Metric Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.xlim((0, NUM_EPOCHS))

    # Plot the title for the X-Axis for the Number of Epochs,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.xlabel("Epoch no.")

    # Plot the title for the Y-Axis for the Loss Values',
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.ylabel("{} Value".format(metric))

    # Plot the legend for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.legend(legends, loc="upper right", frameon=True)

    # Add a Grid to the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.grid(color="white", linestyle="--", linewidth=0.8)

    # Save the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.savefig("./files/images/plots/multi-classes-classification/comparison-optimisers/"
                    "{}-{}-values-plot-all-optimisers-{}.png"
                    .format(subset.lower(), metric.lower(), now_date_time_format))

    # If it is supposed to show the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    if is_to_show:
        # Show the final Plot for the Loss Values' Comparison,
        # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
        # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
        py_plot.show()

    # Close the final Plot for the Loss Values' Comparison,
    # for the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    py_plot.close()


# If the boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs) is set to True
if TENSORFLOW_KERAS_HPC_BACKEND_SESSION:

    # Print the information about if the Model will be executed,
    # using High-Performance Computing (with CPUs and GPUs)
    print("\n")
    print("It will be used High-Performance Computing (with CPUs and GPUs):")
    print(" - Num. CPUS: ", NUM_CPU_PROCESSORS_CORES)
    print(" - Num. GPUS: ", NUM_GPU_DEVICES)
    print("\n")

    # Set the Configuration's Proto, for the given number of Devices (CPUs and GPUs)
    configuration_proto = \
        tensorflow.compat.v1.ConfigProto(device_count={"CPU": NUM_CPU_PROCESSORS_CORES,
                                                       "GPU": NUM_GPU_DEVICES})

    # Configure a TensorFlow Session for High-Performance Computing (with CPUs and GPUs)
    session = tensorflow.compat.v1.Session(config=configuration_proto)

    # Set the current Keras' Backend, with previously configured
    # TensorFlow Session for High-Performance Computing (with CPUs and GPUs)
    keras_backend.set_session(session)


# Retrieve all the Datasets related to the Pokemons' Data:
# i) 4000 examples for the initial Training Set, split in:
#    - 3500 examples for the final Training Set;
#    - 500 examples for the final Validation Set;
# ii) 500 examples for the initial and final Testing Set;
xs_features_training_set_pokemon, xs_masks_training_set_pokemon, \
    ys_classes_training_set_pokemon, ys_labels_training_set_pokemon, \
    xs_features_validation_set_pokemon, xs_masks_validation_set_pokemon, \
    ys_classes_validation_set_pokemon, ys_labels_validation_set_pokemon, \
    xs_features_testing_set_pokemon, xs_masks_testing_set_pokemon, \
    ys_classes_testing_set_pokemon, ys_labels_testing_set_pokemon = \
    retrieve_datasets_from_pokemon_data()

# Create the Images' Data Generator for Pre-Processing with Data Augmentation,
# for the Training Set of the Multi-Classes Problem, in Image Classification
multi_classes_training_image_data_generator_for_preprocessing_with_data_augmentation = \
    image_data_generator_for_preprocessing_with_data_augmentation()

# Generates random Batches of Augmented Data of
# the Images' Data Generator for Pre-Processing with Data Augmentation,
# for the Training Set of the Multi-Classes Problem, in Image Classification
multi_classes_training_set_pokemon_data_augmentation_generator = \
    multi_classes_training_image_data_generator_for_preprocessing_with_data_augmentation \
    .flow(x=xs_features_training_set_pokemon, batch_size=BATCH_SIZE,
          y=ys_classes_training_set_pokemon, shuffle=True)

# Create the Images' Data Generator for Pre-Processing with Data Augmentation,
# for the Validation Set of the Multi-Classes Problem, in Image Classification
multi_classes_validation_image_data_generator_for_preprocessing_with_data_augmentation = \
    image_data_generator_for_preprocessing_with_data_augmentation()

# Generates random Batches of Augmented Data of
# the Images' Data Generator for Pre-Processing with Data Augmentation,
# for the Training Set of the Multi-Classes Problem, in Image Classification
multi_classes_validation_set_pokemon_data_augmentation_generator = \
    multi_classes_validation_image_data_generator_for_preprocessing_with_data_augmentation \
    .flow(x=xs_features_validation_set_pokemon, batch_size=BATCH_SIZE,
          y=ys_classes_validation_set_pokemon, shuffle=True)

# Create the need Early Stopping Callbacks for
# the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification
pokemon_training_loss_early_stopping_callback, \
    pokemon_training_accuracy_early_stopping_callback, \
    pokemon_validation_loss_early_stopping_callback, \
    pokemon_validation_accuracy_early_stopping_callback = \
    create_early_stopping_callbacks()

# Create a list for the Training Losses for all the Optimisers used
optimisers_training_loss_history = []

# Create a list for the Training Accuracies for all the Optimisers used
optimisers_training_accuracy_history = []

# Create a list for the Validation Losses for all the Optimisers used
optimisers_validation_loss_history = []

# Create a list for the Validation Accuracies for all the Optimisers used
optimisers_validation_accuracy_history = []

# Create a list for the Training Losses' Means (Averages) for all the Optimisers used
optimisers_training_loss_means = []

# Create a list for the Training Accuracies' Means (Averages) for all the Optimisers used
optimisers_training_accuracy_means = []

# Create a list for the Validation Losses' Means (Averages) for all the Optimisers used
optimisers_validation_loss_means = []

# Create a list for the Validation Accuracies' Means (Averages) for all the Optimisers used
optimisers_validation_accuracy_means = []

# Create a list for the True/Testing Losses' Means (Averages) for all the Optimisers used
optimisers_true_testing_loss_means = []

# Create a list for the True/Testing Accuracies' Means (Averages) for all the Optimisers used
optimisers_true_testing_accuracy_means = []

# For each Optimiser available
for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

    # Print the initial information line
    print("--------- START OF THE EXECUTION FOR THE {} OPTIMISER ---------"
          .format(AVAILABLE_OPTIMISERS_LIST[num_optimiser]))

    # Retrieve the current DateTime, as custom format
    now_date_time = date_time.utcnow().strftime("%Y%m%d%H%M%S")

    # Set the Root Directory for the Logs of the TensorBoard and TensorFlow
    root_logs_directory = "logs"

    # Set the specific Log Directory,
    # # according to the current executing Optimiser and the current Date and Time (timestamp)
    logs_directory = "{}/model-multi-classes-{}-optimiser-{}/" \
        .format(root_logs_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time)

    # Set the Root Directory for the Weights of the TensorBoard and TensorFlow
    root_weights_directory = "files/weights"

    # Set the specified Sub-Directory, according to the Metrics
    # for the Logs of the TensorBoard and TensorFlow
    file_writer = tensorflow.summary.create_file_writer(logs_directory)

    # Set the File Writer, with previously specified Sub-Directory, according to the Metrics
    # for the Logs of the TensorBoard and TensorFlow
    file_writer.set_as_default()

    # Create the TensorBoard Callback for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.)
    tensorboard_callback = TensorBoard(log_dir=logs_directory)

    # Create the Optimiser to be used for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    current_optimiser = create_optimiser(AVAILABLE_OPTIMISERS_LIST[num_optimiser])

    # Create a Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    cnn_model_in_keras_sequential_api_for_image_classification_multi_classes_classification = \
        create_cnn_model_in_keras_sequential_api_for_image_classification(AVAILABLE_OPTIMISERS_LIST[num_optimiser])

    # Compile the Model for the feed-forward Convolution Neural Network (C.N.N.),
    # with the given Categorical Cross Entropy Loss/Error Function and
    # the Stochastic Gradient Descent (S.G.D.) Optimiser
    cnn_model_in_keras_sequential_api_for_image_classification_multi_classes_classification \
        .compile(loss="categorical_crossentropy",
                 optimizer=current_optimiser,
                 metrics=["accuracy"])

    # Print the Log for the Fitting/Training of
    # the Model for the feed-forward Convolution Neural Network (C.N.N.)
    print(f"\nFitting/Training the Model for "
          f"the feed-forward Convolution Neural Network (C.N.N.) for {NUM_EPOCHS} Epochs "
          f"with a Batch Size of {BATCH_SIZE} and\nan Initial Learning Rate of "
          f"{INITIAL_LEARNING_RATES[num_optimiser]}...\n")

    # Train/Fit the Model for the feed-forward Convolution Neural Network (C.N.N.) for the given NUM_EPOCHS,
    # with the Training Set for the Training Data and the Validation Set for the Validation Data
    cnn_model_in_keras_sequential_api_for_image_classification_training_history = \
        cnn_model_in_keras_sequential_api_for_image_classification_multi_classes_classification \
        .fit(multi_classes_training_set_pokemon_data_augmentation_generator.x,
             multi_classes_training_set_pokemon_data_augmentation_generator.y,
             steps_per_epoch=(NUM_EXAMPLES_FINAL_TRAINING_SET // BATCH_SIZE),
             epochs=NUM_EPOCHS,
             validation_data=(multi_classes_validation_set_pokemon_data_augmentation_generator.x,
                              multi_classes_validation_set_pokemon_data_augmentation_generator.y),
             validation_steps=(NUM_EXAMPLES_FINAL_VALIDATION_SET // BATCH_SIZE),
             batch_size=BATCH_SIZE,
             callbacks=[pokemon_training_loss_early_stopping_callback,
                        pokemon_training_accuracy_early_stopping_callback,
                        pokemon_validation_loss_early_stopping_callback,
                        pokemon_validation_accuracy_early_stopping_callback,
                        tensorboard_callback])

    # the use of High-Performance Computing (with CPUs and GPUs) is set to True
    if TENSORFLOW_KERAS_HPC_BACKEND_SESSION:
        # Clear the current session of the Keras' Backend
        keras_backend.clear_session()

    # Print the final Log for the Fitting/Training of
    # the Model for the feed-forward Convolution Neural Network (C.N.N.)
    print("\nThe Fitting/Training of the Model for "
          "the feed-forward Convolution Neural Network (C.N.N.) is complete!!!\n")

    # Plot the Training's and Validation's Losses,
    # from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    plot_training_and_validation_losses_multi_classes_problem(
        cnn_model_in_keras_sequential_api_for_image_classification_training_history,
        AVAILABLE_OPTIMISERS_LIST[num_optimiser], now_date_time
    )

    # Plot the Training's and Validation's Accuracies,
    # from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
    plot_training_and_validation_accuracies_multi_classes_problem(
        cnn_model_in_keras_sequential_api_for_image_classification_training_history,
        AVAILABLE_OPTIMISERS_LIST[num_optimiser], now_date_time
    )

    # Retrieve the History of the Training Losses for the current Optimiser
    optimiser_training_loss_history = \
        cnn_model_in_keras_sequential_api_for_image_classification_training_history.history["loss"]

    # Retrieve the Number of Epochs History of the Training Losses for the current Optimiser
    num_epochs_optimiser_training_loss_history = len(optimiser_training_loss_history)

    # Store the History of the Training Losses for the current Optimiser,
    # to the list for the Training Losses for all the Optimisers used
    optimisers_training_loss_history \
        .append(optimiser_training_loss_history)

    # Retrieve the History of the Training Accuracies for the current Optimiser
    optimiser_training_accuracy_history = \
        cnn_model_in_keras_sequential_api_for_image_classification_training_history.history["accuracy"]

    # Retrieve the Number of Epochs History of the Training Accuracies for the current Optimiser
    num_epochs_optimiser_training_accuracy_history = len(optimiser_training_accuracy_history)

    # Store the History of the Training Accuracies for the current Optimiser,
    # to the list for the Training Losses for all the Optimisers used
    optimisers_training_accuracy_history \
        .append(optimiser_training_accuracy_history)

    # Retrieve the History of the Validation Losses for the current Optimiser
    optimiser_validation_loss_history = \
        cnn_model_in_keras_sequential_api_for_image_classification_training_history.history["val_loss"]

    # Retrieve the Number of Epochs History of the Validation Losses for the current Optimiser
    num_epochs_optimiser_validation_loss_history = len(optimiser_validation_loss_history)

    # Store the History of the Validation Losses for the current Optimiser,
    # to the list for the Validation Losses for all the Optimisers used
    optimisers_validation_loss_history \
        .append(optimiser_validation_loss_history)

    # Retrieve the History of the Validation Accuracies for the current Optimiser
    optimiser_validation_accuracy_history = \
        cnn_model_in_keras_sequential_api_for_image_classification_training_history.history["val_accuracy"]

    # Retrieve the Number of Epochs History of the Validation Accuracies for the current Optimiser
    num_epochs_optimiser_validation_accuracy_history = len(optimiser_validation_accuracy_history)

    # Store the History of the Validation Accuracies for the current Optimiser,
    # to the list for the Validation Losses for all the Optimisers used
    optimisers_validation_accuracy_history \
        .append(optimiser_validation_accuracy_history)

    # Output the Summary of the architecture of
    # the Model for the feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    cnn_model_in_keras_sequential_api_for_image_classification_multi_classes_classification.summary()

    # Save the Weights of the Neurons of the Fitting/Training of
    # the Model for the feed-forward Convolution Neural Network (C.N.N.)
    cnn_model_in_keras_sequential_api_for_image_classification_multi_classes_classification \
        .save_weights("{}/pokemon-image-classification-training-history-multi-classes-{}-optimiser{}-weights.h5"
                      .format(root_weights_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time))

    # Convert the Model for the feed-forward Convolution Neural Network (C.N.N.) to a JSON Object
    cnn_model_json_object = \
        cnn_model_in_keras_sequential_api_for_image_classification_multi_classes_classification.to_json()

    # Write the Model for the feed-forward Convolution Neural Network (C.N.N.) to a JSON Object
    with open("{}/pokemon-image-classification-training-history-multi-classes-{}-optimiser{}-weights.json"
              .format(root_weights_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time),
              "w") as json_file:

        # Write the JSON Object
        json_file.write(cnn_model_json_object)

    # Predict the Probabilities of Classes for the Testing Set,
    # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
    # fitted/trained previously with the Training and Validation Sets
    ys_classes_testing_set_pokemon_predicted = \
        cnn_model_in_keras_sequential_api_for_image_classification_multi_classes_classification \
        .predict(x=xs_features_testing_set_pokemon,
                 batch_size=BATCH_SIZE, verbose=1)

    # Retrieve the Categorical Cross-Entropy for the Classes' Predictions on the Testing Set,
    # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
    # fitted/trained previously with the Training and Validation Sets
    true_testing_loss = \
        categorical_crossentropy(ys_classes_testing_set_pokemon,
                                 ys_classes_testing_set_pokemon_predicted)

    # Retrieve the Categorical Accuracy for the Classes' Predictions on the Testing Set,
    # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
    # fitted/trained previously with the Training and Validation Sets
    true_testing_accuracy = \
        categorical_accuracy(ys_classes_testing_set_pokemon,
                             ys_classes_testing_set_pokemon_predicted)

    # Just print a blank line, for a better and clearer presentation of the results
    print("\n")

    # Compute the Mean (Average) of the Training Loss, on the Training Set
    training_loss_mean = \
        optimiser_training_loss_history[(num_epochs_optimiser_training_loss_history - 1)]

    # Store the Mean (Average) of the Training Loss, on the Training Set, for the current Optimiser
    optimisers_training_loss_means.append(training_loss_mean)

    # Print the Mean (Average) of the Training Loss, on the Training Set
    print("Training Loss Mean (Average): ", training_loss_mean)

    # Compute the Mean (Average) of the Training Accuracy, on the Training Set
    training_accuracy_mean = \
        optimiser_training_accuracy_history[(num_epochs_optimiser_training_accuracy_history - 1)]

    # Store the Mean (Average) of the Training Accuracy, on the Training Set, for the current Optimiser
    optimisers_training_accuracy_means.append(training_accuracy_mean)

    # Print the Mean (Average) of the Training Accuracy, on the Training Set
    print("Training Accuracy (Average): ", training_accuracy_mean)

    # Compute the Mean (Average) of the Validation Loss, on the Validation Set
    validation_loss_mean = \
        optimiser_validation_loss_history[(num_epochs_optimiser_validation_loss_history - 1)]

    # Store the Mean (Average) of the Validation Loss, on the Validation Set, for the current Optimiser
    optimisers_validation_loss_means.append(validation_loss_mean)

    # Print the Mean (Average) of the Validation Loss, on the Validation Set
    print("Validation Loss Mean (Average): ", validation_loss_mean)

    # Compute the Mean (Average) of the Validation Accuracy, on the Validation Set
    validation_accuracy_mean = \
        optimiser_validation_accuracy_history[(num_epochs_optimiser_validation_accuracy_history - 1)]

    # Store the Mean (Average) of the Validation Accuracy, on the Validation Set, for the current Optimiser
    optimisers_validation_accuracy_means.append(validation_accuracy_mean)

    # Print the Mean (Average) of the Validation Accuracy, on the Validation Set
    print("Validation Accuracy (Average): ", validation_accuracy_mean)

    # Compute the Mean (Average) of the True/Test Loss, on the Testing Set
    true_testing_loss_mean = mean(true_testing_loss)

    # Store the Mean (Average) of the True/Test Loss, on the Testing Set, for the current Optimiser
    optimisers_true_testing_loss_means.append(true_testing_loss_mean)

    # Print the Mean (Average) of the True/Test Loss, on the Testing Set
    print("True/Test Loss Mean (Average): ", true_testing_loss_mean)

    # Compute the Mean (Average) of the True/Test Accuracy, on the Testing Set
    true_testing_accuracy_mean = mean(true_testing_accuracy)

    # Store the Mean (Average) of the True/Test Accuracy, on the Testing Set, for the current Optimiser
    optimisers_true_testing_accuracy_means.append(true_testing_accuracy_mean)

    # Print the Mean (Average) of the True/Test Accuracy, on the Testing Set
    print("True/Test Accuracy Mean (Average): ", true_testing_accuracy_mean)

    # the use of High-Performance Computing (with CPUs and GPUs) is set to True
    if TENSORFLOW_KERAS_HPC_BACKEND_SESSION:
        # Clear the current session of the Keras' Backend
        keras_backend.clear_session()

    # Print the final information line
    print("\n--------- END OF EXECUTION FOR THE {} OPTIMISER ---------\n\n"
          .format(AVAILABLE_OPTIMISERS_LIST[num_optimiser]))


# Retrieve the current DateTime, as custom format
now_date_time = date_time.utcnow().strftime("%Y%m%d%H%M%S")

# Plot the Training Loss Values for all the Optimisers
plot_subset_metric_multi_classes_problem_all_optimisers(optimisers_training_loss_history,
                                                        "Training", "Loss", now_date_time)

# Plot the Training Accuracy Values for all the Optimisers
plot_subset_metric_multi_classes_problem_all_optimisers(optimisers_training_accuracy_history,
                                                        "Training", "Accuracy", now_date_time)

# Plot the Validation Loss Values for all the Optimisers
plot_subset_metric_multi_classes_problem_all_optimisers(optimisers_validation_loss_history,
                                                        "Validation", "Loss", now_date_time)

# Plot the Validation Accuracy Values for all the Optimisers
plot_subset_metric_multi_classes_problem_all_optimisers(optimisers_validation_accuracy_history,
                                                        "Validation", "Accuracy", now_date_time)


# Print the Heading Information about the Losses and Accuracies on the Testing Set
print("------  Final Results for the Losses and Accuracies on "
      "the Testing Set,\nregarding the several Optimisers available ------\n")

# For each Optimiser available
for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

    # Print the respective Means (Averages) for the Losses and Accuracies
    # of the predictions made by the current Optimiser on the Testing Set
    print(" - {}: [ train_loss = {} ; train_acc = {} |"
          " val_loss = {} ; val_acc = {} |"
          " test_loss = {} ; test_acc = {} ]"
          .format(AVAILABLE_OPTIMISERS_LIST[num_optimiser],
                  optimisers_training_loss_means[num_optimiser],
                  optimisers_training_accuracy_means[num_optimiser],
                  optimisers_validation_loss_means[num_optimiser],
                  optimisers_validation_accuracy_means[num_optimiser],
                  optimisers_true_testing_loss_means[num_optimiser],
                  optimisers_true_testing_accuracy_means[num_optimiser]))
