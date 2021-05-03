#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Deep Learning - Assignment #1:
- Pokemon Image Classification and Semantic Segmentation with Deep Neural Networks

Integrated Master of Computer Science and Engineering

NOVA School of Science and Technology,
NOVA University of Lisbon - 2020/2021

Authors:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt) - Student no. 49067
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt) - Student no. 42648

Instructor(s):
- Ludwig Krippahl (ludi@fct.unl.pt)
- Claudia Soares (claudia.soares@fct.unl.pt)

Module for the Semantic Segmentation Problem in the Project

"""

# Import Python's Modules, Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
from project.tp1.libs.images_set_generator import generate_images_sets

operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Warnings from Python's Library
import warnings

# Ignore all the Warnings
# (necessary to ignore the CustomMaskWarning,
#  from the Keras Functional API/TensorFlow Python's Library)
warnings.filterwarnings("ignore")

# Import Mean (Average) Function from the NumPy Python's Library
from numpy import mean

# Import Date Time from Date Time Python's System Module
from datetime import datetime as date_time

# Import the Tensorflow as Tensorflow alias Python's Module
import tensorflow as tensorflow

# Import the Backend Module from the TensorFlow.Python.Keras Python's Module
from tensorflow.python.keras import backend as keras_backend

# Import the Input the TensorFlow.Keras Python's Module
from tensorflow.keras import Input

# Import the Model from the TensorFlow.Keras Python's Module,
# as FunctionalModel alias
from tensorflow.keras import Model as FunctionalModel

# Import the Convolution 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Conv2D

# Import the Batch Normalization Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import BatchNormalization

# Import the Activation Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Activation

# Import the Separable Convolution 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import SeparableConv2D

# Import the Maximum Pooling 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import MaxPooling2D

# Import the Convolution 2D Transpose Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Conv2DTranspose

# Import the Up Sampling 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import UpSampling2D

# Import the Add Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import add

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

# Import the Binary Categorical Cross-Entropy from
# the TensorFlow.Keras.Metrics Python's Module
from tensorflow.keras.metrics import binary_crossentropy

# Import the Binary Categorical Accuracy from
# the TensorFlow.Keras.Metrics Python's Module
from tensorflow.keras.metrics import binary_accuracy

# Import the boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import TENSORFLOW_KERAS_HPC_BACKEND_SESSION

# Import the Number of CPU's Processors/Cores
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_CPU_PROCESSORS_CORES

# Import the Number of GPU's Devices
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_GPU_DEVICES

# Import the Number of Examples (Images/Masks) for the final Training Set
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_EXAMPLES_FINAL_TRAINING_SET

# Import the Number of Examples (Images/Masks) for the final Validation Set
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_EXAMPLES_FINAL_VALIDATION_SET

# Import the Number of Filters for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import IMAGES_HEIGHT

# Import the Number of Filters for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import IMAGES_WIDTH

# Import the Number of Filters for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_CHANNELS_RGB

# Import the Number of Filters for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_CHANNELS_GRAY_SCALE

# Import the Number of Filters for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_FILTERS_PER_BLOCK

# Import the Height of the Kernel of the Filters, for RGB Colors,
# # used for the Model of the feed-forward Convolution Neural Network (C.N.N.)
# # from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import KERNEL_RGB_HEIGHT

# Import the Width of the Kernel of the Filters, for RGB Colors,
# used for the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import KERNEL_RGB_WIDTH

# Import the Height of the Kernel of the Filters, for Gray Scale Colors,
# # used for the Model of the feed-forward Convolution Neural Network (C.N.N.)
# # from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import KERNEL_GRAY_SCALE_HEIGHT

# Import the Width of the Kernel of the Filters, for Gray Scale Colors,
# used for the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import KERNEL_GRAY_SCALE_WIDTH

# Import the Height of the Pooling Matrix #2 used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import POOLING_HEIGHT_2

# Import the Width of the Pooling Matrix #2 used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import POOLING_WIDTH_2

# Import the Height of the Stride used on
# the Pooling Matrices used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import STRIDE_HEIGHT

# Import the Width of the Stride used on
# the Pooling Matrices used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import STRIDE_WIDTH

# Import the Height of the Tuple Sampling used on
# the Pooling Matrices used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import TUPLE_SAMPLING_HEIGHT

# Import the Width of the Tuple Sampling used on
# the Pooling Matrices used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import TUPLE_SAMPLING_WIDTH

# Import the Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import AVAILABLE_OPTIMISERS_LIST

# Import the Number of Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_AVAILABLE_OPTIMISERS

# Import the Learning Rates for the Optimisers used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import INITIAL_LEARNING_RATES

# Import the Momentum #1 for the Optimisers used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import MOMENTUM_1

# Import the Momentum #2 for the Optimiser used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import MOMENTUM_2

# Import the Decay #1 for the Optimisers used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import DECAY_1

# Import the Number of Epochs #2 for the Optimiser for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_EPOCHS_2

# Import the Size of the Batch for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import BATCH_SIZE_1

# Import the function to Retrieve the Datasets from the Pokemons' Data,
# in order to be used to build the model for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.preprocessing_utils import retrieve_datasets_from_pokemon_data

# Import the function to create the Images' Data Generator for
# Pre-Processing with Data Augmentation
# from the Pre-Processing Utils Python's Custom Module
from project.tp1.libs.preprocessing_utils import \
    image_data_generator_for_preprocessing_with_data_augmentation

# Import the function to plot the Training's and Validation's Losses,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Masking, for the Semantic Segmentation Problem
from project.tp1.libs.visualization_plotting import \
    plot_training_and_validation_losses

# Import the function to plot the Training's and Validation's Accuracies,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Masking, for the Semantic Segmentation Problem
from project.tp1.libs.visualization_plotting import \
    plot_training_and_validation_accuracies

# Import the function to plot the Training's and Validation's Losses,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Masking, for the Semantic Segmentation Problem
from project.tp1.libs.visualization_plotting import \
    plot_subset_metric_all_optimisers

# Import the Auxiliary Function to
# Compare the True Testing Masks with the Predicted Masks,
# for the Project, from the TP1_Utils' Python's Module,
# with the compare_true_and_predicted_masks alias
from project.tp1.tp1_utils import compare_masks as compare_true_and_predicted_masks

# Import the Auxiliary Function to
# Overlay the True Testing Masks with the Predicted Masks,
# for the Project, from the TP1_Utils' Python's Module,
# with the overlay_true_and_predicted_masks alias
from project.tp1.tp1_utils import overlay_masks as overlay_true_and_predicted_masks

# Import the Auxiliary Function to
# to convert Images to Pictures,
# for the Project, from the TP1_Utils' Python's Module,
# with the overlay_true_and_predicted_masks alias
from project.tp1.tp1_utils import images_to_pic as images_to_pic


# Function to create the need Early Stopping Callbacks for
# the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Semantic Segmentation
def create_early_stopping_callbacks():

    # Create the Callback for Early Stopping, related to
    # the Loss Cost of the Fitting/Training with Training Set
    training_loss_early_stopping_callback = \
        EarlyStopping(
            monitor='loss',
            min_delta=1e-6,
            patience=NUM_EPOCHS_2,
            verbose=1,
            mode='min',
            baseline=0.08,
            restore_best_weights=True
        )

    # Create the Callback for Early Stopping, related to
    # the Accuracy of the Fitting/Training with Training Set
    training_accuracy_early_stopping_callback = \
        EarlyStopping(
            monitor='binary_accuracy',
            min_delta=1e-6,
            patience=NUM_EPOCHS_2,
            verbose=1,
            mode='max',
            baseline=0.96,
            restore_best_weights=True
        )

    # Create the Callback for Early Stopping, related to
    # the Loss Cost of the Fitting/Training with Validation Set
    validation_loss_early_stopping_callback = \
        EarlyStopping(
            monitor='val_loss',
            min_delta=1e-6,
            patience=NUM_EPOCHS_2,
            verbose=1,
            mode='min',
            baseline=0.08,
            restore_best_weights=True
        )

    # Create the Callback for Early Stopping, related to
    # the Accuracy of the Fitting/Training with Validation Set
    validation_accuracy_early_stopping_callback = \
        EarlyStopping(
            monitor='val_binary_accuracy',
            min_delta=1e-6,
            patience=NUM_EPOCHS_2,
            verbose=1,
            mode='max',
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


# Function to create a Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Semantic Segmentation
def create_cnn_model_in_keras_functional_api_for_semantic_segmentation():

    # Set the Input Layer for the xs (features) of the Pokemon's Data Inputs
    xs_inputs_layer = Input(shape=(IMAGES_HEIGHT, IMAGES_WIDTH) + (NUM_CHANNELS_RGB, ))

    # --- 1st Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 1st Convolution 2D Layer, for the Input features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), using 32 Filters of a Kernel 3x3,
    # Same Padding and an Input Shape of (64 x 64 pixels), as also,
    # 3 Input Dimensions (for each Color Channel - RGB Color)
    xs_features_layer = Conv2D(NUM_FILTERS_PER_BLOCK[0],
                               kernel_size=(KERNEL_RGB_HEIGHT, KERNEL_RGB_WIDTH),
                               strides=(STRIDE_HEIGHT, STRIDE_WIDTH),
                               padding='same')(xs_inputs_layer)

    # Add a Batch Normalization Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    xs_features_layer = BatchNormalization()(xs_features_layer)

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    xs_features_layer = Activation('relu')(xs_features_layer)

    # Keep the current Layer, for the next projected Residual Layer
    previous_block_activation_for_residual_projection = xs_features_layer

    # For each Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.)
    for current_block_id in range(1, 4):

        # --- nth Block of Layers for the Model for
        # the feed-forward Convolution Neural Network (C.N.N.) ---

        # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = Activation('relu')(xs_features_layer)

        # Add a 1st Separable Convolution 2D Layer,
        # for the previous features of the Data/Images of
        # the Pokemons' Dataset given to the Model of the feed-forward
        # Convolution Neural Network (C.N.N.), resulted from the previous layer,
        # using 64 Filters of a Kernel 3x3 and Same Padding
        xs_features_layer = SeparableConv2D(NUM_FILTERS_PER_BLOCK[current_block_id],
                                            kernel_size=(KERNEL_RGB_HEIGHT, KERNEL_RGB_WIDTH),
                                            kernel_initializer='random_uniform',
                                            padding='same')(xs_features_layer)

        # Add a Batch Normalization Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = BatchNormalization()(xs_features_layer)

        # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = Activation('relu')(xs_features_layer)

        # Add a 2nd Separable Convolution 2D Layer,
        # for the previous features of the Data/Images of
        # the Pokemons' Dataset given to the Model of the feed-forward
        # Convolution Neural Network (C.N.N.), resulted from the previous layer,
        # using 64 Filters of a Kernel 3x3 and Same Padding
        xs_features_layer = SeparableConv2D(NUM_FILTERS_PER_BLOCK[current_block_id],
                                            kernel_size=(KERNEL_RGB_HEIGHT, KERNEL_RGB_WIDTH),
                                            kernel_initializer='random_uniform',
                                            padding='same')(xs_features_layer)

        # Add a Batch Normalization Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = BatchNormalization()(xs_features_layer)

        # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
        # with a 2x2 Pooling Size and 2x2 Stride Size
        xs_features_layer = MaxPooling2D((POOLING_HEIGHT_2, POOLING_WIDTH_2),
                                         strides=(STRIDE_HEIGHT, STRIDE_WIDTH),
                                         padding='same')(xs_features_layer)

        # Project the Residual Layer, in Gray Scale Colors, through a Convolution 2D Layer,
        # on the Model of the feed-forward Convolution Neural Network (C.N.N.)
        residual_xs_features_layer_projected = \
            Conv2D(
                NUM_FILTERS_PER_BLOCK[current_block_id],
                kernel_size=(KERNEL_GRAY_SCALE_HEIGHT, KERNEL_GRAY_SCALE_WIDTH),
                kernel_initializer='random_uniform', strides=(STRIDE_HEIGHT, STRIDE_WIDTH),
                padding='same')(
                    previous_block_activation_for_residual_projection
                )

        # Add back the projected Residual Layer, in Gray Scale Colors,
        # to the features of the Data/Images of the Pokemons resulted from
        # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
        xs_features_layer = add([xs_features_layer, residual_xs_features_layer_projected])

        # Keep the current Layer, for the next projected Residual Layer
        previous_block_activation_for_residual_projection = xs_features_layer

    # For each reversed Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.)
    for current_reversed_block_id in range(4):

        # --- nth Block of Layers for the Model for
        # the feed-forward Convolution Neural Network (C.N.N.) ---

        # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = Activation('relu')(xs_features_layer)

        # Add a 1st Convolution Transpose 2D Layer,
        # for the previous features of the Data/Images of
        # the Pokemons' Dataset given to the Model of the feed-forward
        # Convolution Neural Network (C.N.N.), resulted from the previous layer,
        # using 64 Filters of a Kernel 3x3 and Same Padding
        xs_features_layer = Conv2DTranspose(NUM_FILTERS_PER_BLOCK[(3 - current_reversed_block_id)],
                                            kernel_size=(KERNEL_RGB_HEIGHT, KERNEL_RGB_WIDTH),
                                            kernel_initializer='random_uniform',
                                            padding='same')(xs_features_layer)

        # Add a Batch Normalization Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = BatchNormalization()(xs_features_layer)

        # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = Activation('relu')(xs_features_layer)

        # Add a 2nd Convolution Transpose 2D Layer,
        # for the previous features of the Data/Images of
        # the Pokemons' Dataset given to the Model of the feed-forward
        # Convolution Neural Network (C.N.N.), resulted from the previous layer,
        # using 64 Filters of a Kernel 3x3 and Same Padding
        xs_features_layer = Conv2DTranspose(NUM_FILTERS_PER_BLOCK[(3 - current_reversed_block_id)],
                                            kernel_size=(KERNEL_RGB_HEIGHT, KERNEL_RGB_WIDTH),
                                            kernel_initializer='random_uniform',
                                            padding='same')(xs_features_layer)

        # Add a Batch Normalization Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = BatchNormalization()(xs_features_layer)

        # Add an Up Sampling 2D Layer,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        xs_features_layer = \
            UpSampling2D(size=(TUPLE_SAMPLING_HEIGHT, TUPLE_SAMPLING_WIDTH))(xs_features_layer)

        # Project the Residual Layer, through an Up Sampling 2D Layer,
        # on the Model of the feed-forward Convolution Neural Network (C.N.N.)
        residual_xs_features_layer_projected = \
            UpSampling2D(size=(TUPLE_SAMPLING_HEIGHT, TUPLE_SAMPLING_WIDTH))(
                previous_block_activation_for_residual_projection
            )

        # Project the Residual Layer, in Gray Scale Colors,
        # through a Convolutional 2D Layer,
        # on the Model of the feed-forward Convolution Neural Network (C.N.N.)
        residual_xs_features_layer_projected = \
            Conv2D(NUM_FILTERS_PER_BLOCK[(3 - current_reversed_block_id)],
                   kernel_size=(KERNEL_GRAY_SCALE_HEIGHT, KERNEL_GRAY_SCALE_WIDTH),
                   kernel_initializer='random_uniform',
                   padding="same")(residual_xs_features_layer_projected)

        # Add back the projected Residual Layer, in Gray Scale Colors,
        # to the features of the Data/Images of the Pokemons resulted from
        # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
        xs_features_layer = add([xs_features_layer, residual_xs_features_layer_projected])

        # Keep the current Layer, for the previous Block Activation projected as a Residual Layer
        previous_block_activation_for_residual_projection = xs_features_layer

    # Add a last Convolution 2D Layer,
    # for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 64 Filters of a Kernel 3x3 and Same Padding
    xs_features_layer = Conv2D(NUM_CHANNELS_GRAY_SCALE,
                               kernel_size=(KERNEL_RGB_HEIGHT, KERNEL_RGB_WIDTH),
                               kernel_initializer='random_uniform',
                               padding='same')(xs_features_layer)

    # Add the last Sigmoid as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    ys_outputs_layer = Activation('sigmoid')(xs_features_layer)

    # Create a Model for a feed-forward Convolution Neural Network (C.N.N.),
    # which is most appropriate for this type of problem (i.e., Semantic Segmentation),
    # using the Tensorflow Keras' Functional API
    convolution_neural_network_tensorflow_keras_functional_model = \
        FunctionalModel(inputs=xs_inputs_layer, outputs=ys_outputs_layer,
                        name='pokemon-images-semantic-segmentation')

    # Return the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Semantic Segmentation Problem
    return convolution_neural_network_tensorflow_keras_functional_model


# Function to create the Optimiser to be used for
# the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Semantic Segmentation
def create_optimiser(optimiser_id):

    # Initialise the Optimiser to be used for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Semantic Segmentation
    optimiser = None

    # It is being used the Stochastic Gradient Descent (S.G.D.) Optimiser
    if optimiser_id == AVAILABLE_OPTIMISERS_LIST[0]:

        # Initialise the Stochastic Gradient Descent (S.G.D.) Optimiser,
        # with the Learning Rate of 0.5% and Momentum of 90%
        optimiser = SGD(learning_rate=INITIAL_LEARNING_RATES[0],
                        momentum=MOMENTUM_1, decay=DECAY_1)

    # It is being used the Root Mean Squared Prop (R.M.S. PROP) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[1]:

        # Initialise the Root Mean Squared Prop (R.M.S. PROP) Optimiser,
        # with the Learning Rate of 0.5% and Momentum of 90%
        optimiser = RMSprop(learning_rate=INITIAL_LEARNING_RATES[1], momentum=MOMENTUM_2)

    # It is being used the ADAptive Moment estimation (ADA.M.) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[2]:

        # Initialise the ADAptive Moment estimation (ADA.M.) Optimiser,
        # with the Learning Rate of 0.5%
        optimiser = Adam(learning_rate=INITIAL_LEARNING_RATES[2],
                         decay=DECAY_1)

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
    # for the Pokemons' Data, in Semantic Segmentation
    return optimiser


# Function to execute the Model of Semantic Segmentation for all the Available Optimisers
def execute_model_of_semantic_segmentation_for_all_available_optimisers(final_choice=None):

    # If the boolean flag, to keep information about
    # the use of High-Performance Computing (with CPUs and GPUs) is set to True
    if TENSORFLOW_KERAS_HPC_BACKEND_SESSION:

        # Print the information about if the Model will be executed,
        # using High-Performance Computing (with CPUs and GPUs)
        print('\n')
        print('It will be used High-Performance Computing (with CPUs and GPUs):')
        print(' - Num. CPUS: ', NUM_CPU_PROCESSORS_CORES)
        print(' - Num. GPUS: ', NUM_GPU_DEVICES)
        print('\n')

        # Set the Configuration's Proto, for the given number of Devices (CPUs and GPUs)
        configuration_proto = \
            tensorflow.compat.v1.ConfigProto(device_count={'CPU': NUM_CPU_PROCESSORS_CORES,
                                                           'GPU': NUM_GPU_DEVICES})

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
    xs_features_training_set_pokemon, ys_masks_training_set_pokemon, \
        ys_classes_training_set_pokemon, ys_labels_training_set_pokemon, \
        xs_features_validation_set_pokemon, ys_masks_validation_set_pokemon, \
        ys_classes_validation_set_pokemon, ys_labels_validation_set_pokemon, \
        xs_features_testing_set_pokemon, ys_masks_testing_set_pokemon, \
        ys_classes_testing_set_pokemon, ys_labels_testing_set_pokemon = \
        retrieve_datasets_from_pokemon_data()

    # Create the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the xs (features) of the Training Set of the Semantic Segmentation Problem
    features_training_image_data_generator_for_preprocessing_with_data_augmentation = \
        image_data_generator_for_preprocessing_with_data_augmentation()

    # Generates random Batches of Augmented Data of
    # the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the xs (features) of the Training Set of the Semantic Segmentation Problem
    features_training_set_pokemon_data_augmentation_generator = \
        features_training_image_data_generator_for_preprocessing_with_data_augmentation \
        .flow(x=xs_features_training_set_pokemon, batch_size=BATCH_SIZE_1,
              y=ys_classes_training_set_pokemon, shuffle=True)

    # Create the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the xs (masks) of the Training Set of the Semantic Segmentation Problem
    masks_training_image_data_generator_for_preprocessing_with_data_augmentation = \
        image_data_generator_for_preprocessing_with_data_augmentation()

    # Generates random Batches of Augmented Data of
    # the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the xs (masks) of the Training Set of the Semantic Segmentation Problem
    masks_training_set_pokemon_data_augmentation_generator = \
        masks_training_image_data_generator_for_preprocessing_with_data_augmentation \
        .flow(x=ys_masks_training_set_pokemon, batch_size=BATCH_SIZE_1,
              y=ys_classes_training_set_pokemon, shuffle=True)

    # Create the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the xs (features) of the Validation Set of the Semantic Segmentation Problem
    features_validation_image_data_generator_for_preprocessing_with_data_augmentation = \
        image_data_generator_for_preprocessing_with_data_augmentation()

    # Generates random Batches of Augmented Data of
    # the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the xs (features) of the Validation Set of the Semantic Segmentation Problem
    features_validation_set_pokemon_data_augmentation_generator = \
        features_validation_image_data_generator_for_preprocessing_with_data_augmentation \
        .flow(x=xs_features_validation_set_pokemon, batch_size=BATCH_SIZE_1,
              y=ys_classes_validation_set_pokemon, shuffle=True)

    # Create the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the xs (masks) of the Validation Set of the Semantic Segmentation Problem
    masks_validation_image_data_generator_for_preprocessing_with_data_augmentation = \
        image_data_generator_for_preprocessing_with_data_augmentation()

    # Generates random Batches of Augmented Data of
    # the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the xs (masks) of the Validation Set of the Semantic Segmentation Problem
    masks_validation_set_pokemon_data_augmentation_generator = \
        masks_validation_image_data_generator_for_preprocessing_with_data_augmentation \
        .flow(x=ys_masks_validation_set_pokemon, batch_size=BATCH_SIZE_1,
              y=ys_classes_validation_set_pokemon, shuffle=True)

    # Generate Images' Set, for the several sets
    # (i.e., Training, Validation and Testing Sets)
    generate_images_sets(xs_features_training_set_pokemon, ys_masks_training_set_pokemon,
                         features_training_set_pokemon_data_augmentation_generator.x,
                         masks_training_set_pokemon_data_augmentation_generator.x,
                         xs_features_validation_set_pokemon, ys_masks_validation_set_pokemon,
                         features_validation_set_pokemon_data_augmentation_generator.x,
                         masks_validation_set_pokemon_data_augmentation_generator.x,
                         xs_features_testing_set_pokemon, ys_masks_testing_set_pokemon)

    # Create the need Early Stopping Callbacks for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Semantic Segmentation
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

        # Only execute the final choice for the Optimiser
        if (AVAILABLE_OPTIMISERS_LIST[num_optimiser] == final_choice) and (final_choice is not None):

            # Print the initial information line
            print('--------- START OF THE EXECUTION FOR THE %s OPTIMISER ---------'
                  % (AVAILABLE_OPTIMISERS_LIST[num_optimiser]))

            # Retrieve the current DateTime, as custom format
            now_date_time = date_time.utcnow().strftime('%Y%m%d%H%M%S')

            # Set the Root Directory for the Logs of the TensorBoard and TensorFlow
            root_logs_directory = 'logs'

            # Set the specific Log Directory,
            # # according to the current executing Optimiser and the current Date and Time (timestamp)
            logs_directory = '%s\\model-semantic-segmentation-%s-optimiser-%s\\' \
                             % (root_logs_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time)

            # Set the Root Directory for the Weights of the TensorBoard and TensorFlow
            root_weights_directory = 'files\\weights'

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
            # for the Pokemons' Data, in Semantic Segmentation
            current_optimiser = create_optimiser(AVAILABLE_OPTIMISERS_LIST[num_optimiser])

            # Create a Model for a feed-forward Convolution Neural Network (C.N.N.),
            # for the Pokemons' Data, in Semantic Segmentation
            cnn_model_in_keras_functional_api_for_semantic_segmentation_masking = \
                create_cnn_model_in_keras_functional_api_for_semantic_segmentation()

            # Compile the Model for the feed-forward Convolution Neural Network (C.N.N.),
            # with the given Categorical Cross Entropy Loss/Error Function and
            # the Stochastic Gradient Descent (S.G.D.) Optimiser
            cnn_model_in_keras_functional_api_for_semantic_segmentation_masking \
                .compile(loss='binary_crossentropy',
                         optimizer=current_optimiser,
                         metrics=['binary_accuracy'])

            # Print the Log for the Fitting/Training of
            # the Model for the feed-forward Convolution Neural Network (C.N.N.)
            print(f'\nFitting/Training the Model for '
                  f'the feed-forward Convolution Neural Network (C.N.N.) for {NUM_EPOCHS_2} Epochs '
                  f'with a Batch Size of {BATCH_SIZE_1} and\nan Initial Learning Rate of '
                  f'{INITIAL_LEARNING_RATES[num_optimiser]}...\n')

            # Train/Fit the Model for the feed-forward Convolution Neural Network (C.N.N.) for the given NUM_EPOCHS,
            # with the Training Set for the Training Data and the Validation Set for the Validation Data
            cnn_model_in_keras_functional_api_for_semantic_segmentation_training_history = \
                cnn_model_in_keras_functional_api_for_semantic_segmentation_masking \
                .fit(features_training_set_pokemon_data_augmentation_generator.x,
                     masks_training_set_pokemon_data_augmentation_generator.x,
                     steps_per_epoch=(NUM_EXAMPLES_FINAL_TRAINING_SET // BATCH_SIZE_1),
                     epochs=NUM_EPOCHS_2,
                     validation_data=(features_validation_set_pokemon_data_augmentation_generator.x,
                                      masks_validation_set_pokemon_data_augmentation_generator.x),
                     validation_steps=(NUM_EXAMPLES_FINAL_VALIDATION_SET // BATCH_SIZE_1),
                     batch_size=BATCH_SIZE_1,
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
            print('\nThe Fitting/Training of the Model for '
                  'the feed-forward Convolution Neural Network (C.N.N.) is complete!!!\n')

            # Plot the Training's and Validation's Losses,
            # from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
            # for the Pokemons' Data, in Image Classification, for the Multi-Labels Problem
            plot_training_and_validation_losses(
                cnn_model_in_keras_functional_api_for_semantic_segmentation_training_history,
                AVAILABLE_OPTIMISERS_LIST[num_optimiser], now_date_time, 'Semantic-Segmentation'
            )

            # Plot the Training's and Validation's Accuracies,
            # from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
            # for the Pokemons' Data, in Image Classification, for the Multi-Labels Problem
            plot_training_and_validation_accuracies(
                cnn_model_in_keras_functional_api_for_semantic_segmentation_training_history,
                AVAILABLE_OPTIMISERS_LIST[num_optimiser], now_date_time, 'Semantic-Segmentation'
            )

            # Retrieve the History of the Training Losses for the current Optimiser
            optimiser_training_loss_history = \
                cnn_model_in_keras_functional_api_for_semantic_segmentation_training_history.history['loss']

            # Retrieve the Number of Epochs History of the Training Losses for the current Optimiser
            num_epochs_optimiser_training_loss_history = len(optimiser_training_loss_history)

            # Store the History of the Training Losses for the current Optimiser,
            # to the list for the Training Losses for all the Optimisers used
            optimisers_training_loss_history \
                .append(optimiser_training_loss_history)

            # Retrieve the History of the Training Accuracies for the current Optimiser
            optimiser_training_accuracy_history = \
                cnn_model_in_keras_functional_api_for_semantic_segmentation_training_history.history['binary_accuracy']

            # Retrieve the Number of Epochs History of the Training Accuracies for the current Optimiser
            num_epochs_optimiser_training_accuracy_history = len(optimiser_training_accuracy_history)

            # Store the History of the Training Accuracies for the current Optimiser,
            # to the list for the Training Losses for all the Optimisers used
            optimisers_training_accuracy_history \
                .append(optimiser_training_accuracy_history)

            # Retrieve the History of the Validation Losses for the current Optimiser
            optimiser_validation_loss_history = \
                cnn_model_in_keras_functional_api_for_semantic_segmentation_training_history.history['val_loss']

            # Retrieve the Number of Epochs History of the Validation Losses for the current Optimiser
            num_epochs_optimiser_validation_loss_history = len(optimiser_validation_loss_history)

            # Store the History of the Validation Losses for the current Optimiser,
            # to the list for the Validation Losses for all the Optimisers used
            optimisers_validation_loss_history \
                .append(optimiser_validation_loss_history)

            # Retrieve the History of the Validation Accuracies for the current Optimiser
            optimiser_validation_accuracy_history = \
                cnn_model_in_keras_functional_api_for_semantic_segmentation_training_history\
                .history['val_binary_accuracy']

            # Retrieve the Number of Epochs History of the Validation Accuracies for the current Optimiser
            num_epochs_optimiser_validation_accuracy_history = len(optimiser_validation_accuracy_history)

            # Store the History of the Validation Accuracies for the current Optimiser,
            # to the list for the Validation Losses for all the Optimisers used
            optimisers_validation_accuracy_history \
                .append(optimiser_validation_accuracy_history)

            # Output the Summary of the architecture of
            # the Model for the feed-forward Convolution Neural Network (C.N.N.),
            # for the Pokemons' Data, in Semantic Segmentation
            cnn_model_in_keras_functional_api_for_semantic_segmentation_masking.summary()

            # Save the Weights of the Neurons of the Fitting/Training of
            # the Model for the feed-forward Convolution Neural Network (C.N.N.)
            cnn_model_in_keras_functional_api_for_semantic_segmentation_masking \
                .save_weights('%s/pokemon-semantic-segmentation-training-history-%s-optimiser-%s-weights.h5'
                              % (root_weights_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(),
                                 now_date_time))

            # Convert the Model for the feed-forward Convolution Neural Network (C.N.N.) to a JSON Object
            cnn_model_json_object = \
                cnn_model_in_keras_functional_api_for_semantic_segmentation_masking.to_json()

            # Write the Model for the feed-forward Convolution Neural Network (C.N.N.) to a JSON Object
            with open('%s/pokemon-semantic-segmentation-training-history-%s-optimiser-%s-weights.json'
                      % (root_weights_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time),
                      'w') as json_file:

                # Write the JSON Object
                json_file.write(cnn_model_json_object)

            # Predict the Masks for the Testing Set,
            # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
            # fitted/trained previously with the Training and Validation Sets
            ys_masks_testing_set_pokemon_predicted = \
                cnn_model_in_keras_functional_api_for_semantic_segmentation_masking \
                .predict(x=xs_features_testing_set_pokemon,
                         batch_size=BATCH_SIZE_1, verbose=1)

            # Convert the Images of the predicted Masks for the Testing Set to Pictures
            images_to_pic(('files\\images\\figures\\testing\\predictions\\masks\\'
                          'pokemon-semantic-segmentation-masks-%s-optimiser.png'
                           % (AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower())),
                          ys_masks_testing_set_pokemon_predicted)

            # Retrieve the Binary Cross-Entropy for the Masks' Predictions on the Testing Set,
            # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
            # fitted/trained previously with the Training and Validation Sets
            true_testing_loss = \
                binary_crossentropy(ys_masks_testing_set_pokemon,
                                    ys_masks_testing_set_pokemon_predicted)

            # Retrieve the Binary Accuracy for the Masks' Predictions on the Testing Set,
            # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
            # fitted/trained previously with the Training and Validation Sets
            true_testing_accuracy = \
                binary_accuracy(ys_masks_testing_set_pokemon,
                                ys_masks_testing_set_pokemon_predicted)

            # Create the Image to Compare the True and Predicted Masks, in Image Masking/Semantic Segmentation
            compare_true_and_predicted_masks(
                ('files\\images\\figures\\testing\\predictions\\masks-comparison\\'
                 'pokemon-semantic-segmentation-masks-comparison-%s-optimiser.png'
                 % (AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower())),
                ys_masks_testing_set_pokemon, ys_masks_testing_set_pokemon_predicted
            )

            # Create the Image to Overlay the True and Predicted Masks, in Image Masking/Semantic Segmentation
            overlay_true_and_predicted_masks(
                ('files\\images\\figures\\testing\\predictions\\masks-overlay\\'
                 'pokemon-semantic-segmentation-masks-overlay-%s-optimiser.png'
                 % (AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower())),
                xs_features_testing_set_pokemon, ys_masks_testing_set_pokemon_predicted
            )

            # Just print a blank line, for a better and clearer presentation of the results
            print('\n')

            # Compute the Mean (Average) of the Training Loss, on the Training Set
            training_loss_mean = \
                optimiser_training_loss_history[(num_epochs_optimiser_training_loss_history - 1)]

            # Store the Mean (Average) of the Training Loss, on the Training Set, for the current Optimiser
            optimisers_training_loss_means.append(training_loss_mean)

            # Print the Mean (Average) of the Training Loss, on the Training Set
            print('Training Loss Mean (Average): ', training_loss_mean)

            # Compute the Mean (Average) of the Training Accuracy, on the Training Set
            training_accuracy_mean = \
                optimiser_training_accuracy_history[(num_epochs_optimiser_training_accuracy_history - 1)]

            # Store the Mean (Average) of the Training Accuracy, on the Training Set, for the current Optimiser
            optimisers_training_accuracy_means.append(training_accuracy_mean)

            # Print the Mean (Average) of the Training Accuracy, on the Training Set
            print('Training Accuracy (Average): ', training_accuracy_mean)

            # Compute the Mean (Average) of the Validation Loss, on the Validation Set
            validation_loss_mean = \
                optimiser_validation_loss_history[(num_epochs_optimiser_validation_loss_history - 1)]

            # Store the Mean (Average) of the Validation Loss, on the Validation Set, for the current Optimiser
            optimisers_validation_loss_means.append(validation_loss_mean)

            # Print the Mean (Average) of the Validation Loss, on the Validation Set
            print('Validation Loss Mean (Average): ', validation_loss_mean)

            # Compute the Mean (Average) of the Validation Accuracy, on the Validation Set
            validation_accuracy_mean = \
                optimiser_validation_accuracy_history[(num_epochs_optimiser_validation_accuracy_history - 1)]

            # Store the Mean (Average) of the Validation Accuracy, on the Validation Set, for the current Optimiser
            optimisers_validation_accuracy_means.append(validation_accuracy_mean)

            # Print the Mean (Average) of the Validation Accuracy, on the Validation Set
            print('Validation Accuracy (Average): ', validation_accuracy_mean)

            # Compute the Mean (Average) of the True/Test Loss, on the Testing Set
            true_testing_loss_mean = mean(true_testing_loss)

            # Store the Mean (Average) of the True/Test Loss, on the Testing Set, for the current Optimiser
            optimisers_true_testing_loss_means.append(true_testing_loss_mean)

            # Print the Mean (Average) of the True/Test Loss, on the Testing Set
            print('True/Test Loss Mean (Average): ', true_testing_loss_mean)

            # Compute the Mean (Average) of the True/Test Accuracy, on the Testing Set
            true_testing_accuracy_mean = mean(true_testing_accuracy)

            # Store the Mean (Average) of the True/Test Accuracy, on the Testing Set, for the current Optimiser
            optimisers_true_testing_accuracy_means.append(true_testing_accuracy_mean)

            # Print the Mean (Average) of the True/Test Accuracy, on the Testing Set
            print('True/Test Accuracy Mean (Average): ', true_testing_accuracy_mean)

            # the use of High-Performance Computing (with CPUs and GPUs) is set to True
            if TENSORFLOW_KERAS_HPC_BACKEND_SESSION:

                # Clear the current session of the Keras' Backend
                keras_backend.clear_session()

            # Print the final information line
            print('\n--------- END OF EXECUTION FOR THE %s OPTIMISER ---------\n\n'
                  % (AVAILABLE_OPTIMISERS_LIST[num_optimiser]))

    # Retrieve the current DateTime, as custom format
    now_date_time = date_time.utcnow().strftime('%Y%m%d%H%M%S')

    # Plot the Training Loss Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_training_loss_history,
                                      'Training', 'Loss', now_date_time,
                                      'Semantic-Segmentation', final_choice=final_choice)

    # Plot the Training Accuracy Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_training_accuracy_history,
                                      'Training', 'Accuracy', now_date_time,
                                      'Semantic-Segmentation', final_choice=final_choice)

    # Plot the Validation Loss Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_validation_loss_history,
                                      'Validation', 'Loss', now_date_time,
                                      'Semantic-Segmentation', final_choice=final_choice)

    # Plot the Validation Accuracy Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_validation_accuracy_history,
                                      'Validation', 'Accuracy', now_date_time,
                                      'Semantic-Segmentation', final_choice=final_choice)

    # Print the Heading Information about the Losses and Accuracies on the Testing Set
    print('------  Final Results for the Losses and Accuracies on '
          'the Testing Set,\nregarding the several Optimisers available ------\n')

    # For each Optimiser available
    for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

        # Only execute the final choice for the Optimiser
        if (AVAILABLE_OPTIMISERS_LIST[num_optimiser] == final_choice) and (final_choice is not None):

            # Print the respective Means (Averages) for the Losses and Accuracies
            # of the predictions made by the current Optimiser on the Training, Validation and Testing Set
            print(' - %s: [ train_loss = %.12f ; train_binary_acc = %.12f |'
                  ' val_loss = %.12f ; val_binary_acc = %.12f |'
                  ' test_loss = %.12f ; test_binary_acc = %.12f ]'
                  % (AVAILABLE_OPTIMISERS_LIST[num_optimiser],
                     optimisers_training_loss_means[0],
                     optimisers_training_accuracy_means[0],
                     optimisers_validation_loss_means[0],
                     optimisers_validation_accuracy_means[0],
                     optimisers_true_testing_loss_means[0],
                     optimisers_true_testing_accuracy_means[0]))

        # Execute all the Optimisers
        elif final_choice is None:

            # Print the respective Means (Averages) for the Losses and Accuracies
            # of the predictions made by the current Optimiser on the Training, Validation and Testing Set
            print(' - %s: [ train_loss = %.12f ; train_binary_acc = %.12f |'
                  ' val_loss = %.12f ; val_binary_acc = %.12f |'
                  ' test_loss = %.12f ; test_binary_acc = %.12f ]'
                  % (AVAILABLE_OPTIMISERS_LIST[num_optimiser],
                     optimisers_training_loss_means[num_optimiser],
                     optimisers_training_accuracy_means[num_optimiser],
                     optimisers_validation_loss_means[num_optimiser],
                     optimisers_validation_accuracy_means[num_optimiser],
                     optimisers_true_testing_loss_means[num_optimiser],
                     optimisers_true_testing_accuracy_means[num_optimiser]))

    # Print two break lines, for a better presentation of the output
    print('\n\n')

    # Return the respective Means (Averages) for the Losses and Accuracies
    # of the predictions made by the current Optimiser on the Training, Validation and Testing Set
    return optimisers_training_loss_means, \
        optimisers_training_accuracy_means, \
        optimisers_validation_loss_means, \
        optimisers_validation_accuracy_means, \
        optimisers_true_testing_loss_means, \
        optimisers_true_testing_accuracy_means
