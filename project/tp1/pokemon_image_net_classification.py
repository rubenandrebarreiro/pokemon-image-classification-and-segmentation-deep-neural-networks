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

Module for the Multi-Class and Multi-Label Classification Problem,
with the Pre-Trained MobileNet Model, using the Weights of the ImageNet, in the Project

"""

# Import Python's Modules, Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Import the Logging Library as logging
import logging as logging

# Disable all the Warnings from the Logging Library
logging.disable(logging.WARNING)

# Disable all the Debugging Logs from TensorFlow Library
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

# Import the Model from the TensorFlow.Keras Python's Module,
# as FunctionalModel alias
from tensorflow.keras import Model as FunctionalModel

# Import the MobileNet from the TensorFlow.Keras.Applications.MobileNet Python's Module
from tensorflow.keras.applications.mobilenet import MobileNet

# Import the Global Average Pooling 2D Layer from
# the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import GlobalAveragePooling2D

# Import the Dense Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Dense

# Import the Activation Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Activation

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

# Import the Reduce Learning Rate on Plateau from
# the TensorFlow.Keras.Callbacks Python's Module
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Import the TensorBoard from
# the TensorFlow.Keras.Callbacks Python's Module
from tensorflow.keras.callbacks import TensorBoard

# Import the Categorical Cross-Entropy from
# the TensorFlow.Keras.Metrics Python's Module
from tensorflow.keras.metrics import categorical_crossentropy

# Import the Categorical Accuracy from
# the TensorFlow.Keras.Metrics Python's Module
from tensorflow.keras.metrics import categorical_accuracy

# Import the Binary Cross-Entropy from
# the TensorFlow.Keras.Metrics Python's Module
from tensorflow.keras.metrics import binary_crossentropy

# Import the Binary Accuracy from
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

# Import the Number of RGB Channels for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_CHANNELS_RGB

# Import the Number of Units of the last Dense Layer #2 for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_UNITS_LAST_DENSE_LAYER_2

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

# Import the Decay #2 for the Optimiser used for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import DECAY_2

# Import the Number of Epochs for the Optimiser for
# the Model of the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_EPOCHS

# Import the Size of the Batch for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import BATCH_SIZE_2

# Import the Number of Classes for the Datasets from the Pokemons' Data
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_CLASSES_POKEMON_TYPES

# Import the function to Retrieve the Datasets from the Pokemons' Data,
# in order to be used to build the model for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.preprocessing_utils import retrieve_datasets_from_pokemon_data

# Import the function to create the Images' Data Generator for Pre-Processing with Data Augmentation
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.preprocessing_utils import \
    image_data_generator_for_preprocessing_with_data_augmentation

# Import the function to plot the Training's and Validation's Losses,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
from project.tp1.libs.visualization_plotting import \
    plot_training_and_validation_losses

# Import the function to plot the Training's and Validation's Accuracies,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
from project.tp1.libs.visualization_plotting import \
    plot_training_and_validation_accuracies

# Import the function to plot the Training's and Validation's Losses,
# from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
from project.tp1.libs.visualization_plotting import \
    plot_subset_metric_all_optimisers


# Function to create the need Early Stopping Callbacks for
# the Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Multi-Class and Multi-Label Problem
def create_early_stopping_callbacks(classification_problem):

    # Initialise the Early Stopping Callbacks for the Training and Validation Accuracies
    training_accuracy_early_stopping_callback = validation_accuracy_early_stopping_callback = None

    # Create the Callback for Early Stopping, related to
    # the Loss Cost of the Fitting/Training with Training Set
    training_loss_early_stopping_callback = \
        EarlyStopping(
            monitor='loss',
            min_delta=1e-6,
            patience=NUM_EPOCHS,
            verbose=1,
            mode='min',
            baseline=0.12,
            restore_best_weights=True
        )

    # If it is the Multi-Class Classification Problem
    if classification_problem.lower() == 'multi-class':

        # Create the Callback for Early Stopping, related to
        # the Accuracy of the Fitting/Training with Training Set
        training_accuracy_early_stopping_callback = \
            EarlyStopping(
                monitor='accuracy',
                min_delta=1e-6,
                patience=NUM_EPOCHS,
                verbose=1,
                mode='max',
                baseline=0.94,
                restore_best_weights=True
            )

    # If it is the Multi-Label Classification Problem
    elif classification_problem.lower() == 'multi-label':

        # Create the Callback for Early Stopping, related to
        # the Accuracy of the Fitting/Training with Training Set
        training_accuracy_early_stopping_callback = \
            EarlyStopping(
                monitor='binary_accuracy',
                min_delta=1e-6,
                patience=NUM_EPOCHS,
                verbose=1,
                mode='max',
                baseline=0.94,
                restore_best_weights=True
            )

    # Create the Callback for Early Stopping, related to
    # the Loss Cost of the Fitting/Training with Validation Set
    validation_loss_early_stopping_callback = \
        EarlyStopping(
            monitor='val_loss',
            min_delta=1e-6,
            patience=NUM_EPOCHS,
            verbose=1,
            mode='min',
            baseline=0.16,
            restore_best_weights=True
        )

    # If it is the Multi-Class Classification Problem
    if classification_problem.lower() == 'multi-class':

        # Create the Callback for Early Stopping, related to
        # the Accuracy of the Fitting/Training with Validation Set
        validation_accuracy_early_stopping_callback = \
            EarlyStopping(
                monitor='val_accuracy',
                min_delta=1e-6,
                patience=NUM_EPOCHS,
                verbose=1,
                mode='max',
                baseline=0.92,
                restore_best_weights=True
            )

    # If it is the Multi-Label Classification Problem
    elif classification_problem.lower() == 'multi-label':

        # Create the Callback for Early Stopping, related to
        # the Accuracy of the Fitting/Training with Validation Set
        validation_accuracy_early_stopping_callback = \
            EarlyStopping(
                monitor='val_binary_accuracy',
                min_delta=1e-6,
                patience=NUM_EPOCHS,
                verbose=1,
                mode='max',
                baseline=0.92,
                restore_best_weights=True
            )

    # Return need Early Stopping Callbacks for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    return training_loss_early_stopping_callback, \
        training_accuracy_early_stopping_callback, \
        validation_loss_early_stopping_callback, \
        validation_accuracy_early_stopping_callback


# Function to create a Fine Tuned Model, using the Layers of
# the MobileNet Model and the Weights of the ImageNet Dataset,
# for a feed-forward Convolution Neural Network (C.N.N.), for the Pokemons' Data, in Image Classification
def create_fine_tuned_mobile_net_model_in_keras_functional_api_for_image_classification(classification_problem):

    # Create the Base Model, using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    mobile_net_base_model = \
        MobileNet(weights='imagenet', include_top=False,
                  input_shape=(IMAGES_HEIGHT, IMAGES_WIDTH, NUM_CHANNELS_RGB))

    # If it is Multi-Class Classification Problem
    if classification_problem.lower() == 'multi-class':

        # Set the first 20 Layers of the Pre-Trained MobileNet Model, as not Trainable (i.e., freeze them)
        for mobile_net_large_base_model_layer in mobile_net_base_model.layers[:20]:

            # Set the current Layer of the MobileNet Model, as not Trainable (i.e., freeze it)
            mobile_net_large_base_model_layer.trainable = False

        # Set the remaining Layers of the Pre-Trained MobileNet Model, as Trainable (i.e., unfreeze them)
        for mobile_net_large_base_model_layer in mobile_net_base_model.layers[20:]:

            # Set the current Layer of the MobileNet Model, as Trainable (i.e., do not freeze it)
            mobile_net_large_base_model_layer.trainable = True

    # If it is Multi-Label Classification Problem
    if classification_problem.lower() == 'multi-label':

        # Set the first 50 Layers of the Pre-Trained MobileNet Model, as not Trainable (i.e., freeze them)
        for mobile_net_large_base_model_layer in mobile_net_base_model.layers[:50]:

            # Set the current Layer of the MobileNet Model, as not Trainable (i.e., freeze it)
            mobile_net_large_base_model_layer.trainable = False

        # Set the remaining Layers of the Pre-Trained MobileNet Model, as Trainable (i.e., unfreeze them)
        for mobile_net_large_base_model_layer in mobile_net_base_model.layers[50:]:

            # Set the current Layer of the MobileNet Model, as Trainable (i.e., do not freeze it)
            mobile_net_large_base_model_layer.trainable = True

    # Retrieve the xs (features) from the Input (first layer) of the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_initial_input = mobile_net_base_model.input

    # Retrieve the xs (features) from the Output (last layer) of the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_layer = mobile_net_base_model.output

    # Add a Global Spatial Average Pooling 2D Layer to the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_layer = GlobalAveragePooling2D()(xs_features_layer)

    # Add a Dense Layer with 2048 Units to the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_layer = Dense(NUM_UNITS_LAST_DENSE_LAYER_2, kernel_initializer='he_uniform')(xs_features_layer)

    # Add a ReLU (Rectified Linear Unit) Activation Function Layer to the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_layer = Activation('relu')(xs_features_layer)

    # Add a Dense Layer with 2048 Units to the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_layer = Dense(NUM_UNITS_LAST_DENSE_LAYER_2, kernel_initializer='he_uniform')(xs_features_layer)

    # Add a ReLU (Rectified Linear Unit) Activation Function Layer to the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_layer = Activation('relu')(xs_features_layer)

    # Add a Dense Layer with 10 (Number of Classes) Units to the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_layer = Dense(NUM_CLASSES_POKEMON_TYPES, kernel_initializer='he_uniform')(xs_features_layer)

    # Initialise the xs (features) extracted from the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    xs_features_final_output = None

    # If it is the Multi-Class Classification Problem
    if classification_problem.lower() == 'multi-class':

        # Add a Softmax Activation Function Layer to the Base Model,
        # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
        xs_features_final_output = Activation('softmax')(xs_features_layer)

    # If it is the Multi-Label Classification Problem
    elif classification_problem.lower() == 'multi-label':

        # Add a Softmax Activation Function Layer to the Base Model,
        # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
        xs_features_final_output = Activation('sigmoid')(xs_features_layer)

    # Create the final Fine-Tuned Functional Model, with Input and Outputs of the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    fine_tuned_cnn_model_keras_functional_api = \
        FunctionalModel(name='%s-classification-mobile-net-image-net-weights' % (classification_problem.lower()),
                        inputs=xs_features_initial_input, outputs=xs_features_final_output)

    # Return the final Fine-Tuned Functional Model, with Input and Outputs of the Base Model,
    # using the Layers of the MobileNet Model and the Weights of the ImageNet Dataset
    return fine_tuned_cnn_model_keras_functional_api


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

        # Initialise the Stochastic Gradient Descent (S.G.D.) Optimiser
        optimiser = SGD(learning_rate=INITIAL_LEARNING_RATES[0],
                        momentum=MOMENTUM_1, decay=DECAY_2)

    # It is being used the Root Mean Squared Prop (R.M.S. PROP) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[1]:

        # Initialise the Root Mean Squared Prop (R.M.S. PROP) Optimiser
        optimiser = RMSprop(learning_rate=INITIAL_LEARNING_RATES[1], momentum=MOMENTUM_2)

    # It is being used the ADAptive Moment estimation (ADA.M.) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[2]:

        # Initialise the ADAptive Moment estimation (ADA.M.) Optimiser
        optimiser = Adam(learning_rate=INITIAL_LEARNING_RATES[2],
                         decay=DECAY_1)

    # It is being used the ADAptive GRADient algorithm (ADA.GRAD.) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[3]:

        # Initialise the ADAptive GRADient algorithm (ADA.GRAD.) Optimiser
        optimiser = Adagrad(learning_rate=INITIAL_LEARNING_RATES[3])

    # It is being used the ADAptive DELTA algorithm (ADA.DELTA) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[4]:

        # Initialise the ADAptive DELTA algorithm (ADA.DELTA) Optimiser
        optimiser = Adadelta(learning_rate=INITIAL_LEARNING_RATES[4])

    # It is being used the ADAptive MAX algorithm (ADA.MAX.) Optimiser
    elif optimiser_id == AVAILABLE_OPTIMISERS_LIST[5]:

        # Initialise the ADAptive MAX algorithm (ADA.MAX.) Optimiser
        optimiser = Adamax(learning_rate=INITIAL_LEARNING_RATES[5])

    # Return the Optimiser to be used for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    return optimiser


# Function to execute the MobileNet Model of
# Multi-Class Classification for all the Available Optimisers
def execute_mobile_net_model_multi_class_classification_for_all_available_optimisers():

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
    xs_features_training_set_pokemon, xs_masks_training_set_pokemon, \
        ys_classes_training_set_pokemon, ys_labels_training_set_pokemon, \
        xs_features_validation_set_pokemon, ys_masks_validation_set_pokemon, \
        ys_classes_validation_set_pokemon, ys_labels_validation_set_pokemon, \
        xs_features_testing_set_pokemon, ys_masks_testing_set_pokemon, \
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
        .flow(x=xs_features_training_set_pokemon, y=ys_classes_training_set_pokemon,
              batch_size=BATCH_SIZE_2, shuffle=True)

    # Create the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the Validation Set of the Multi-Classes Problem, in Image Classification
    multi_classes_validation_image_data_generator_for_preprocessing_with_data_augmentation = \
        image_data_generator_for_preprocessing_with_data_augmentation()

    # Generates random Batches of Augmented Data of
    # the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the Training Set of the Multi-Classes Problem, in Image Classification
    multi_classes_validation_set_pokemon_data_augmentation_generator = \
        multi_classes_validation_image_data_generator_for_preprocessing_with_data_augmentation \
        .flow(x=xs_features_validation_set_pokemon, y=ys_classes_validation_set_pokemon,
              batch_size=BATCH_SIZE_2, shuffle=True)

    # Create the need Early Stopping Callbacks for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    pokemon_training_loss_early_stopping_callback, \
        pokemon_training_accuracy_early_stopping_callback, \
        pokemon_validation_loss_early_stopping_callback, \
        pokemon_validation_accuracy_early_stopping_callback = \
        create_early_stopping_callbacks('Multi-Class')

    # Create the need Reduce Learning Rate on Plateau for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    reduce_learning_rate_on_plateau_callback = \
        ReduceLROnPlateau(monitor='val_loss', patience=4, cooldown=2, rate=0.6, min_lr=1e-18, verbose=1)

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
        print('--------- START OF THE EXECUTION FOR THE %s OPTIMISER ---------'
              % (AVAILABLE_OPTIMISERS_LIST[num_optimiser]))

        # Retrieve the current DateTime, as custom format
        now_date_time = date_time.utcnow().strftime('%Y%m%d%H%M%S')

        # Set the Root Directory for the Logs of the TensorBoard and TensorFlow
        root_logs_directory = 'logs'

        # Set the specific Log Directory,
        # according to the current executing Optimiser and the current Date and Time (timestamp)
        logs_directory = '%s\\image-net-pre-trained-multi-classes-%s-optimiser-%s\\' \
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
        # for the Pokemons' Data, in Image Classification
        current_optimiser = create_optimiser(AVAILABLE_OPTIMISERS_LIST[num_optimiser])

        # Create the Fine Tuned Model, using the Layers of
        # the MobileNet Model and the Weights of the ImageNet Dataset,
        # for a feed-forward Convolution Neural Network (C.N.N.), for the Pokemons' Data, in Image Classification
        fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_classes_classification = \
            create_fine_tuned_mobile_net_model_in_keras_functional_api_for_image_classification('multi-class')

        # Compile the Model for the feed-forward Convolution Neural Network (C.N.N.),
        # with the given Categorical Cross Entropy Loss/Error Function and
        # the Stochastic Gradient Descent (S.G.D.) Optimiser
        fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_classes_classification \
            .compile(loss='categorical_crossentropy',
                     optimizer=current_optimiser,
                     metrics=['accuracy'])

        # Print the Log for the Fitting/Training of
        # the Model for the feed-forward Convolution Neural Network (C.N.N.)
        print(f'\nFitting/Training the Model for '
              f'the feed-forward Convolution Neural Network (C.N.N.) for {NUM_EPOCHS} Epochs '
              f'with a Batch Size of {BATCH_SIZE_2} and\nan Initial Learning Rate of '
              f'{INITIAL_LEARNING_RATES[num_optimiser]}...\n')

        # Train/Fit the Model for the feed-forward Convolution Neural Network (C.N.N.) for the given NUM_EPOCHS,
        # with the Training Set for the Training Data and the Validation Set for the Validation Data
        cnn_model_in_keras_sequential_api_for_image_classification_training_history = \
            fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_classes_classification \
            .fit(multi_classes_training_set_pokemon_data_augmentation_generator,
                 steps_per_epoch=(NUM_EXAMPLES_FINAL_TRAINING_SET // BATCH_SIZE_2),
                 epochs=NUM_EPOCHS,
                 validation_data=multi_classes_validation_set_pokemon_data_augmentation_generator,
                 validation_steps=(NUM_EXAMPLES_FINAL_VALIDATION_SET // BATCH_SIZE_2),
                 batch_size=BATCH_SIZE_2,
                 callbacks=[pokemon_training_loss_early_stopping_callback,
                            pokemon_training_accuracy_early_stopping_callback,
                            pokemon_validation_loss_early_stopping_callback,
                            pokemon_validation_accuracy_early_stopping_callback,
                            reduce_learning_rate_on_plateau_callback,
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
        # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
        plot_training_and_validation_losses(
            cnn_model_in_keras_sequential_api_for_image_classification_training_history,
            AVAILABLE_OPTIMISERS_LIST[num_optimiser], now_date_time, 'Multi-Class'
        )

        # Plot the Training's and Validation's Accuracies,
        # from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
        # for the Pokemons' Data, in Image Classification, for the Multi-Class Problem
        plot_training_and_validation_accuracies(
            cnn_model_in_keras_sequential_api_for_image_classification_training_history,
            AVAILABLE_OPTIMISERS_LIST[num_optimiser], now_date_time, 'Multi-Class'
        )

        # Retrieve the History of the Training Losses for the current Optimiser
        optimiser_training_loss_history = \
            cnn_model_in_keras_sequential_api_for_image_classification_training_history.history['loss']

        # Retrieve the Number of Epochs History of the Training Losses for the current Optimiser
        num_epochs_optimiser_training_loss_history = len(optimiser_training_loss_history)

        # Store the History of the Training Losses for the current Optimiser,
        # to the list for the Training Losses for all the Optimisers used
        optimisers_training_loss_history \
            .append(optimiser_training_loss_history)

        # Retrieve the History of the Training Accuracies for the current Optimiser
        optimiser_training_accuracy_history = \
            cnn_model_in_keras_sequential_api_for_image_classification_training_history.history['accuracy']

        # Retrieve the Number of Epochs History of the Training Accuracies for the current Optimiser
        num_epochs_optimiser_training_accuracy_history = len(optimiser_training_accuracy_history)

        # Store the History of the Training Accuracies for the current Optimiser,
        # to the list for the Training Losses for all the Optimisers used
        optimisers_training_accuracy_history \
            .append(optimiser_training_accuracy_history)

        # Retrieve the History of the Validation Losses for the current Optimiser
        optimiser_validation_loss_history = \
            cnn_model_in_keras_sequential_api_for_image_classification_training_history.history['val_loss']

        # Retrieve the Number of Epochs History of the Validation Losses for the current Optimiser
        num_epochs_optimiser_validation_loss_history = len(optimiser_validation_loss_history)

        # Store the History of the Validation Losses for the current Optimiser,
        # to the list for the Validation Losses for all the Optimisers used
        optimisers_validation_loss_history \
            .append(optimiser_validation_loss_history)

        # Retrieve the History of the Validation Accuracies for the current Optimiser
        optimiser_validation_accuracy_history = \
            cnn_model_in_keras_sequential_api_for_image_classification_training_history.history['val_accuracy']

        # Retrieve the Number of Epochs History of the Validation Accuracies for the current Optimiser
        num_epochs_optimiser_validation_accuracy_history = len(optimiser_validation_accuracy_history)

        # Store the History of the Validation Accuracies for the current Optimiser,
        # to the list for the Validation Losses for all the Optimisers used
        optimisers_validation_accuracy_history \
            .append(optimiser_validation_accuracy_history)

        # Output the Summary of the architecture of
        # the Model for the feed-forward Convolution Neural Network (C.N.N.),
        # for the Pokemons' Data, in Image Classification
        fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_classes_classification.summary()

        # Save the Weights of the Neurons of the Fitting/Training of
        # the Model for the feed-forward Convolution Neural Network (C.N.N.)
        fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_classes_classification \
            .save_weights('%s/pokemon-image-classification-training-history-multi-classes-%s-optimiser-'
                          '%s-weights-image-net.h5'
                          % (root_weights_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time))

        # Convert the Model for the feed-forward Convolution Neural Network (C.N.N.) to a JSON Object
        cnn_model_json_object = \
            fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_classes_classification.to_json()

        # Write the Model for the feed-forward Convolution Neural Network (C.N.N.) to a JSON Object
        with open('%s/pokemon-image-classification-training-history-multi-classes-%s-optimiser-'
                  '%s-weights-image-net.json'
                  % (root_weights_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time),
                  'w') as json_file:

            # Write the JSON Object
            json_file.write(cnn_model_json_object)

        # Predict the Probabilities of Classes for the Testing Set,
        # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
        # fitted/trained previously with the Training and Validation Sets
        ys_classes_testing_set_pokemon_predicted = \
            fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_classes_classification \
            .predict(x=xs_features_testing_set_pokemon,
                     batch_size=BATCH_SIZE_2, verbose=1)

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
                                      'Multi-Class', image_net_pre_trained=True)

    # Plot the Training Accuracy Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_training_accuracy_history,
                                      'Training', 'Accuracy', now_date_time,
                                      'Multi-Class', image_net_pre_trained=True)

    # Plot the Validation Loss Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_validation_loss_history,
                                      'Validation', 'Loss', now_date_time,
                                      'Multi-Class', image_net_pre_trained=True)

    # Plot the Validation Accuracy Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_validation_accuracy_history,
                                      'Validation', 'Accuracy', now_date_time,
                                      'Multi-Class', image_net_pre_trained=True)

    # Print the Heading Information about the Losses and Accuracies on the Testing Set
    print('------  Final Results for the Losses and Accuracies on '
          'the Testing Set,\nregarding the several Optimisers available ------\n')

    # For each Optimiser available
    for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

        # Print the respective Means (Averages) for the Losses and Accuracies
        # of the predictions made by the current Optimiser on the Testing Set
        print(' - %s: [ train_loss = %.12f ; train_acc = %.12f |'
              ' val_loss = %.12f ; val_acc = %.12f |'
              ' test_loss = %.12f ; test_acc = %.12f ]'
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


# Function to execute the MobileNet Model of
# Multi-Label Classification for all the Available Optimisers
def execute_mobile_net_model_multi_label_classification_for_all_available_optimisers():

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
    xs_features_training_set_pokemon, xs_masks_training_set_pokemon, \
        ys_classes_training_set_pokemon, ys_labels_training_set_pokemon, \
        xs_features_validation_set_pokemon, ys_masks_validation_set_pokemon, \
        ys_classes_validation_set_pokemon, ys_labels_validation_set_pokemon, \
        xs_features_testing_set_pokemon, ys_masks_testing_set_pokemon, \
        ys_classes_testing_set_pokemon, ys_labels_testing_set_pokemon = \
        retrieve_datasets_from_pokemon_data()

    # Create the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the Training Set of the Multi-Labels Problem, in Image Classification
    multi_labels_training_image_data_generator_for_preprocessing_with_data_augmentation = \
        image_data_generator_for_preprocessing_with_data_augmentation()

    # Generates random Batches of Augmented Data of
    # the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the Training Set of the Multi-Labels Problem, in Image Classification
    multi_labels_training_set_pokemon_data_augmentation_generator = \
        multi_labels_training_image_data_generator_for_preprocessing_with_data_augmentation \
        .flow(x=xs_features_training_set_pokemon, y=ys_labels_training_set_pokemon,
              batch_size=BATCH_SIZE_2, shuffle=True)

    # Create the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the Validation Set of the Multi-Labels Problem, in Image Classification
    multi_labels_validation_image_data_generator_for_preprocessing_with_data_augmentation = \
        image_data_generator_for_preprocessing_with_data_augmentation()

    # Generates random Batches of Augmented Data of
    # the Images' Data Generator for Pre-Processing with Data Augmentation,
    # for the Training Set of the Multi-Labels Problem, in Image Classification
    multi_labels_validation_set_pokemon_data_augmentation_generator = \
        multi_labels_validation_image_data_generator_for_preprocessing_with_data_augmentation \
        .flow(x=xs_features_validation_set_pokemon, y=ys_labels_validation_set_pokemon,
              batch_size=BATCH_SIZE_2, shuffle=True)

    # Create the need Early Stopping Callbacks for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    pokemon_training_loss_early_stopping_callback, \
        pokemon_training_accuracy_early_stopping_callback, \
        pokemon_validation_loss_early_stopping_callback, \
        pokemon_validation_accuracy_early_stopping_callback = \
        create_early_stopping_callbacks('Multi-Label')

    # Create the need Reduce Learning Rate on Plateau for
    # the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification
    reduce_learning_rate_on_plateau_callback = \
        ReduceLROnPlateau(monitor='val_loss', patience=4, cooldown=2, rate=0.6, min_lr=1e-18, verbose=1)

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
        print('--------- START OF THE EXECUTION FOR THE %s OPTIMISER ---------'
              % (AVAILABLE_OPTIMISERS_LIST[num_optimiser]))

        # Retrieve the current DateTime, as custom format
        now_date_time = date_time.utcnow().strftime('%Y%m%d%H%M%S')

        # Set the Root Directory for the Logs of the TensorBoard and TensorFlow
        root_logs_directory = 'logs'

        # Set the specific Log Directory,
        # according to the current executing Optimiser and the current Date and Time (timestamp)
        logs_directory = '%s\\image-net-pre-trained-multi-labels-%s-optimiser-%s\\' \
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
        # for the Pokemons' Data, in Image Classification
        current_optimiser = create_optimiser(AVAILABLE_OPTIMISERS_LIST[num_optimiser])

        # Create the Fine Tuned Model, using the Layers of
        # the MobileNet Model and the Weights of the ImageNet Dataset,
        # for a feed-forward Convolution Neural Network (C.N.N.), for the Pokemons' Data, in Image Classification
        fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_labels_classification = \
            create_fine_tuned_mobile_net_model_in_keras_functional_api_for_image_classification('Multi-Label')

        # Compile the Model for the feed-forward Convolution Neural Network (C.N.N.),
        # with the given Categorical Cross Entropy Loss/Error Function and
        # the Stochastic Gradient Descent (S.G.D.) Optimiser
        fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_labels_classification \
            .compile(loss='binary_crossentropy',
                     optimizer=current_optimiser,
                     metrics=['binary_accuracy'])

        # Print the Log for the Fitting/Training of
        # the Model for the feed-forward Convolution Neural Network (C.N.N.)
        print(f'\nFitting/Training the Model for '
              f'the feed-forward Convolution Neural Network (C.N.N.) for {NUM_EPOCHS} Epochs '
              f'with a Batch Size of {BATCH_SIZE_2} and\nan Initial Learning Rate of '
              f'{INITIAL_LEARNING_RATES[num_optimiser]}...\n')

        # Train/Fit the Model for the feed-forward Convolution Neural Network (C.N.N.) for the given NUM_EPOCHS,
        # with the Training Set for the Training Data and the Validation Set for the Validation Data
        cnn_model_in_keras_sequential_api_for_image_classification_training_history = \
            fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_labels_classification \
            .fit(multi_labels_training_set_pokemon_data_augmentation_generator,
                 steps_per_epoch=(NUM_EXAMPLES_FINAL_TRAINING_SET // BATCH_SIZE_2),
                 epochs=NUM_EPOCHS,
                 validation_data=multi_labels_validation_set_pokemon_data_augmentation_generator,
                 validation_steps=(NUM_EXAMPLES_FINAL_VALIDATION_SET // BATCH_SIZE_2),
                 batch_size=BATCH_SIZE_2,
                 callbacks=[pokemon_training_loss_early_stopping_callback,
                            pokemon_training_accuracy_early_stopping_callback,
                            pokemon_validation_loss_early_stopping_callback,
                            pokemon_validation_accuracy_early_stopping_callback,
                            reduce_learning_rate_on_plateau_callback,
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
        # for the Pokemons' Data, in Image Classification, for the Multi-Label Problem
        plot_training_and_validation_losses(
            cnn_model_in_keras_sequential_api_for_image_classification_training_history,
            AVAILABLE_OPTIMISERS_LIST[num_optimiser], now_date_time, 'Multi-Label'
        )

        # Plot the Training's and Validation's Accuracies,
        # from the History of the Model for a feed-forward Convolution Neural Network (C.N.N.),
        # for the Pokemons' Data, in Image Classification, for the Multi-Label Problem
        plot_training_and_validation_accuracies(
            cnn_model_in_keras_sequential_api_for_image_classification_training_history,
            AVAILABLE_OPTIMISERS_LIST[num_optimiser], now_date_time, 'Multi-Label'
        )

        # Retrieve the History of the Training Losses for the current Optimiser
        optimiser_training_loss_history = \
            cnn_model_in_keras_sequential_api_for_image_classification_training_history.history['loss']

        # Retrieve the Number of Epochs History of the Training Losses for the current Optimiser
        num_epochs_optimiser_training_loss_history = len(optimiser_training_loss_history)

        # Store the History of the Training Losses for the current Optimiser,
        # to the list for the Training Losses for all the Optimisers used
        optimisers_training_loss_history \
            .append(optimiser_training_loss_history)

        # Retrieve the History of the Training Accuracies for the current Optimiser
        optimiser_training_accuracy_history = \
            cnn_model_in_keras_sequential_api_for_image_classification_training_history.history['binary_accuracy']

        # Retrieve the Number of Epochs History of the Training Accuracies for the current Optimiser
        num_epochs_optimiser_training_accuracy_history = len(optimiser_training_accuracy_history)

        # Store the History of the Training Accuracies for the current Optimiser,
        # to the list for the Training Losses for all the Optimisers used
        optimisers_training_accuracy_history \
            .append(optimiser_training_accuracy_history)

        # Retrieve the History of the Validation Losses for the current Optimiser
        optimiser_validation_loss_history = \
            cnn_model_in_keras_sequential_api_for_image_classification_training_history.history['val_loss']

        # Retrieve the Number of Epochs History of the Validation Losses for the current Optimiser
        num_epochs_optimiser_validation_loss_history = len(optimiser_validation_loss_history)

        # Store the History of the Validation Losses for the current Optimiser,
        # to the list for the Validation Losses for all the Optimisers used
        optimisers_validation_loss_history \
            .append(optimiser_validation_loss_history)

        # Retrieve the History of the Validation Accuracies for the current Optimiser
        optimiser_validation_accuracy_history = \
            cnn_model_in_keras_sequential_api_for_image_classification_training_history.history['val_binary_accuracy']

        # Retrieve the Number of Epochs History of the Validation Accuracies for the current Optimiser
        num_epochs_optimiser_validation_accuracy_history = len(optimiser_validation_accuracy_history)

        # Store the History of the Validation Accuracies for the current Optimiser,
        # to the list for the Validation Losses for all the Optimisers used
        optimisers_validation_accuracy_history \
            .append(optimiser_validation_accuracy_history)

        # Output the Summary of the architecture of
        # the Model for the feed-forward Convolution Neural Network (C.N.N.),
        # for the Pokemons' Data, in Image Classification
        fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_labels_classification.summary()

        # Save the Weights of the Neurons of the Fitting/Training of
        # the Model for the feed-forward Convolution Neural Network (C.N.N.)
        fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_labels_classification \
            .save_weights('%s/pokemon-image-classification-training-history-multi-labels-'
                          '%s-optimiser-%s-weights-image-net.h5'
                          % (root_weights_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time))

        # Convert the Model for the feed-forward Convolution Neural Network (C.N.N.) to a JSON Object
        cnn_model_json_object = \
            fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_labels_classification.to_json()

        # Write the Model for the feed-forward Convolution Neural Network (C.N.N.) to a JSON Object
        with open('%s/pokemon-image-classification-training-history-multi-labels-'
                  '%s-optimiser-%s-weights-image-net.json'
                  % (root_weights_directory, AVAILABLE_OPTIMISERS_LIST[num_optimiser].lower(), now_date_time),
                  'w') as json_file:

            # Write the JSON Object
            json_file.write(cnn_model_json_object)

        # Predict the Probabilities of Labels for the Testing Set,
        # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
        # fitted/trained previously with the Training and Validation Sets
        ys_labels_testing_set_pokemon_predicted = \
            fine_tuned_cnn_model_in_keras_functional_api_for_image_classification_multi_labels_classification \
            .predict(x=xs_features_testing_set_pokemon,
                     batch_size=BATCH_SIZE_2, verbose=1)

        # Retrieve the Categorical Cross-Entropy for the Labels' Predictions on the Testing Set,
        # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
        # fitted/trained previously with the Training and Validation Sets
        true_testing_loss = \
            binary_crossentropy(ys_labels_testing_set_pokemon,
                                ys_labels_testing_set_pokemon_predicted)

        # Retrieve the Categorical Accuracy for the Labels' Predictions on the Testing Set,
        # using the Model for the feed-forward Convolution Neural Network (C.N.N.),
        # fitted/trained previously with the Training and Validation Sets
        true_testing_accuracy = \
            binary_accuracy(ys_labels_testing_set_pokemon,
                            ys_labels_testing_set_pokemon_predicted)

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
                                      'Multi-Label', image_net_pre_trained=True)

    # Plot the Training Accuracy Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_training_accuracy_history,
                                      'Training', 'Accuracy', now_date_time,
                                      'Multi-Label', image_net_pre_trained=True)

    # Plot the Validation Loss Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_validation_loss_history,
                                      'Validation', 'Loss', now_date_time,
                                      'Multi-Label', image_net_pre_trained=True)

    # Plot the Validation Accuracy Values for all the Optimisers
    plot_subset_metric_all_optimisers(optimisers_validation_accuracy_history,
                                      'Validation', 'Accuracy', now_date_time,
                                      'Multi-Label', image_net_pre_trained=True)

    # Print the Heading Information about the Losses and Accuracies on the Testing Set
    print('------  Final Results for the Losses and Accuracies on '
          'the Testing Set,\nregarding the several Optimisers available, using the ImageNet Weights ------\n')

    # For each Optimiser available
    for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):
        # Print the respective Means (Averages) for the Losses and Accuracies
        # of the predictions made by the current Optimiser on the Testing Set
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
