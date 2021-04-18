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

# Import Keras from the TensorFlow Python's Library
from tensorflow import keras

# Import the Sequential from the TensorFlow.Keras.Models Python's Module
from tensorflow.keras.models import Sequential

# Import the Convolution 2D Layer from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import Conv2D

# Import the Activation Layer from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import Activation

# Import the Batch Normalization Layer from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import BatchNormalization

# Import the Max Pooling 2D Layer from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import MaxPooling2D

# Import the Dropout Layer from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import Dropout

# Import the Flatten Layer from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import Flatten

# Import the Dense Layer from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import Dense

# Import the Auxiliary Function to Load the Data for the Project, from the TP1_Utils' Module
from tp1_utils import load_data


# Constants

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
# the Convolution Neural Network (C.N.N.)
NUM_FILTERS = [32, 32, 64, 64]

# The Height of the Kernel of the Filters used for the Model of
# the Convolution Neural Network (C.N.N.)
KERNEL_HEIGHT = 3

# The Width of the Kernel of the Filters used for the Model of
# the Convolution Neural Network (C.N.N.)
KERNEL_WIDTH = 3

# The Height of the Pooling Matrix used for the Model of
# the Convolution Neural Network (C.N.N.)
POOLING_HEIGHT = 2

# The Width of the Pooling Matrix used for the Model of
# the Convolution Neural Network (C.N.N.)
POOLING_WIDTH = 2

# The Height of the Stride used on
# the Pooling Matrices used for the Model of
# the Convolution Neural Network (C.N.N.)
STRIDE_HEIGHT = 2

# The Width of the Stride used on
# the Pooling Matrices used for the Model of
# the Convolution Neural Network (C.N.N.)
STRIDE_WIDTH = 2

# The Optimisers available to use for the the Model of
# the Convolution Neural Network (C.N.N.)
AVAILABLE_OPTIMISERS_LIST = ["SGD", "RMSPROP", "ADAM", "ADAGRAD", "ADADELTA"]

# The Learning Rate for the Optimizer used for
# the Model of the Convolution Neural Network (C.N.N.)
INITIAL_LEARNING_RATE = 0.05

# The Number of Epochs for the Optimiser for
# the Model of the Convolution Neural Network (C.N.N.)
NUM_EPOCHS = 1000

# The Size of the Batch for the Model of
# the Convolution Neural Network (C.N.N.)
BATCH_SIZE = 32


# Retrieve the Datasets from the Pokemons' Data,
# in order to be used to build the model for the Model of
# the Convolution Neural Network (C.N.N.)
def retrieve_datasets_from_pokemon_data():

    # Load the Data needed for the Project, using the given Auxiliary Function for it,
    # containing all the Datasets related to the Pokemons' Data
    pokemon_datasets = load_data()

    # Retrieve the xs (features) of the Pokemons' Datasets, for the Training Set
    xs_features_training_set = pokemon_datasets['train_X'][:3500]

    # Retrieve the xs (masks) of the Pokemons' Datasets, for the Training Set
    xs_masks_training_set = pokemon_datasets['train_masks'][:3500]

    # Retrieve the ys (classes) of the Pokemons' Datasets, for the Training Set
    ys_classes_training_set = pokemon_datasets['train_classes'][:3500]

    # Retrieve the ys (labels) of the Pokemons' Datasets, for the Training Set
    ys_labels_training_set = pokemon_datasets['train_labels'][:3500]

    # Retrieve the xs (features) of the Pokemons' Datasets, for the Validation Set
    xs_features_validation_set = pokemon_datasets['train_X'][3500:]

    # Retrieve the xs (masks) of the Pokemons' Datasets, for the Validation Set
    xs_masks_validation_set = pokemon_datasets['train_masks'][3500:]

    # Retrieve the ys (classes) of the Pokemons' Datasets, for the Validation Set
    ys_classes_validation_set = pokemon_datasets['train_classes'][3500:]

    # Retrieve the ys (labels) of the Pokemons' Datasets, for the Validation Set
    ys_labels_validation_set = pokemon_datasets['train_labels'][3500:]

    # Retrieve the xs (features) of the Pokemons' Datasets, for the Testing Set
    xs_features_testing_set = pokemon_datasets['test_X']

    # Retrieve the xs (masks) of the Pokemons' Datasets, for the Testing Set
    xs_masks_testing_set = pokemon_datasets['test_masks']

    # Retrieve the ys (classes) of the Pokemons' Datasets, for the Testing Set
    ys_classes_testing_set = pokemon_datasets['test_classes']

    # Retrieve the ys (labels) of the Pokemons' Datasets, for the Testing Set
    ys_labels_testing_set = pokemon_datasets['test_labels']

    # Return all the retrieved Datasets related to the Pokemons' Data
    return xs_features_training_set, xs_masks_training_set, ys_classes_training_set, ys_labels_training_set,\
        xs_features_validation_set, xs_masks_validation_set, ys_classes_validation_set, ys_labels_validation_set,\
        xs_features_testing_set, xs_masks_testing_set, ys_classes_testing_set, ys_labels_testing_set


# Create a Model for a feed-forward Convolution Neural Network (C.N.N.)
def create_cnn_model_in_keras_sequential_api_for_image_classification():

    # Create a Model for a feed-forward Convolution Neural Network (C.N.N.),
    # which is most appropriate for this type of problem (i.e., Image Classification),
    # using the Tensorflow Keras' Sequential API
    convolution_neural_network_tensorflow_keras_sequential_model = Sequential()

    # 1st CONV => RELU => CONV => RELU => POOL layer set

    # Add a first Convolution 2D Layer, for the Input features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), using 32 Filters of a Kernel 3x3,
    # Same Padding and an Input Shape of (64 x 64 pixels), as also,
    # 3 Input Dimensions (for each Color Channel - RGB Color)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Conv2D(NUM_FILTERS[0], (KERNEL_HEIGHT, KERNEL_WIDTH), padding="same",
             input_shape=(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS_RGB)))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Activation("relu"))

    # Add a Batch Normalization Layer, to normalize the features of
    # the Data/Images of the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # by re-centering and re-scaling that features and making the Model of
    # the feed-forward Convolution Neural Network (C.N.N.) to be faster and more stable
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(BatchNormalization(axis=-1))

    # Add a second Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 32 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Conv2D(NUM_FILTERS[1], (KERNEL_HEIGHT, KERNEL_WIDTH), padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Activation("relu"))

    # Add a Batch Normalization Layer, to normalize the features of
    # the Data/Images of the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # by re-centering and re-scaling that features and making the Model of
    # the feed-forward Convolution Neural Network (C.N.N.) to be faster and more stable
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(BatchNormalization(axis=-1))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # Add a Dropout Layer, for Regularization of the features of
    # the Data/Images of the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # using as hyper-parameter, the Dropout Rate of 25%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network_tensorflow_keras_sequential_model.add(Dropout(0.25))

    # 2nd CONV => RELU => CONV => RELU => POOL layer set

    # Add a third Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 64 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Conv2D(NUM_FILTERS[2], (KERNEL_HEIGHT, KERNEL_WIDTH), padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Activation("relu"))

    # Add a Batch Normalization Layer, to normalize the features of
    # the Data/Images of the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # by re-centering and re-scaling that features and making the Model of
    # the feed-forward Convolution Neural Network (C.N.N.) to be faster and more stable
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(BatchNormalization(axis=-1))

    # Add a fourth Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 64 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Conv2D(NUM_FILTERS[3], (KERNEL_HEIGHT, KERNEL_WIDTH), padding="same"))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Activation("relu"))

    # Add a Batch Normalization Layer, to normalize the features of
    # the Data/Images of the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # by re-centering and re-scaling that features and making the Model of
    # the feed-forward Convolution Neural Network (C.N.N.) to be faster and more stable
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(BatchNormalization(axis=-1))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # Add a Dropout Layer, for Regularization of the features of
    # the Data/Images of the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # using as hyper-parameter, the Dropout Rate of 25%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Dropout(0.25))

    # 1st (and only) set of FC => RELU layers

    # Add a Flatten Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.)
    # NOTE:
    # - This is needed to flatten the input into a single dimension for the features,
    #   which is what the next Dense Layer needs;
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Flatten())

    # Add a Dense Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # for a total of 512 Units (Weights and Biases)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Dense(512))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Activation("relu"))

    # Add a Batch Normalization Layer, to normalize the features of
    # the Data/Images of the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # by re-centering and re-scaling that features and making the Model of
    # the feed-forward Convolution Neural Network (C.N.N.) to be faster and more stable
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(BatchNormalization())

    # Add a Dropout Layer, for Regularization of the features of
    # the Data/Images of the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # using as hyper-parameter, the Dropout Rate of 50%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Dropout(0.5))

    # SoftMax Classifier

    # Add a Dense Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # for a total of 10 Units (Weights and Biases)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Dense(10))

    # Add a Softmax as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Activation("softmax"))

    # Return the Model of the feed-forward Convolution Neural Network (C.N.N.)
    return convolution_neural_network_tensorflow_keras_sequential_model


# Retrieve all the Datasets related to the Pokemons' Data:
# i) 4000 examples for the initial Training Set, split in:
#    - 3500 examples for the final Training Set;
#    - 500 examples for the final Validation Set;
# ii) 500 examples for the initial and final Testing Set;
xs_features_training_set_pokemon, xs_masks_training_set_pokemon,\
    ys_classes_training_set_pokemon, ys_labels_training_set_pokemon,\
    xs_features_validation_set_pokemon, xs_masks_validation_set_pokemon,\
    ys_classes_validation_set_pokemon, ys_labels_validation_set_pokemon,\
    xs_features_testing_set_pokemon, xs_masks_testing_set_pokemon,\
    ys_classes_testing_set_pokemon, ys_labels_testing_set_pokemon = \
    retrieve_datasets_from_pokemon_data()
