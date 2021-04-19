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

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import the Sequential from the TensorFlow.Keras.Models Python's Module
from tensorflow.keras.models import Sequential

# Import the Convolution 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Conv2D

# Import the Activation Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Activation

# Import the Batch Normalization Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import BatchNormalization

# Import the Max Pooling 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import MaxPooling2D

# Import the Dropout Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Dropout

# Import the Flatten Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Flatten

# Import the Dense Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Dense

# Import the Stochastic Gradient Descent (S.G.D.) Optimizer
# from the TensorFlow.Keras.Optimizers Python's Module
from tensorflow.keras.optimizers import SGD

# Import the Root Mean Squared Prop (R.M.S. PROP) Optimizer
# from the TensorFlow.Keras.Optimizers Python's Module
from tensorflow.keras.optimizers import RMSprop

# Import the ADAptive Moment estimation (ADA.M.) Optimizer
# from the TensorFlow.Keras.Optimizers Python's Module
from tensorflow.keras.optimizers import Adam

# Import the ADAptive GRADient algorithm (ADA.GRAD.) Optimizer
# from the TensorFlow.Keras.Optimizers Python's Module
from tensorflow.keras.optimizers import Adagrad

# Import the ADAptive DELTA algorithm (ADA.DELTA) Optimizer
# from the TensorFlow.Keras.Optimizers Python's Module
from tensorflow.keras.optimizers import Adadelta

# Import the ADAptive MAX algorithm (ADA.MAX.) Optimizer
# from the TensorFlow.Keras.Optimizers Python's Module
from tensorflow.keras.optimizers import Adamax

# Import the Auxiliary Function to Load the Data for the Project, from the TP1_Utils' Python's Module
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
NUM_FILTERS = [16, 32, 64, 128]

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
INITIAL_LEARNING_RATE = 0.005

# The Number of Epochs for the Optimiser for
# the Model of the Convolution Neural Network (C.N.N.)
NUM_EPOCHS = 50

# The Size of the Batch for the Model of
# the Convolution Neural Network (C.N.N.)
BATCH_SIZE = 16


# Retrieve the Datasets from the Pokemons' Data,
# in order to be used to build the model for the Model of
# the Convolution Neural Network (C.N.N.)
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
    return xs_features_training_set, xs_masks_training_set, ys_classes_training_set, ys_labels_training_set,\
        xs_features_validation_set, xs_masks_validation_set, ys_classes_validation_set, ys_labels_validation_set,\
        xs_features_testing_set, xs_masks_testing_set, ys_classes_testing_set, ys_labels_testing_set


def image_data_generator_for_preprocessing_with_data_augmentation():

    image_data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    return image_data_generator


# Function to create a Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification
def create_cnn_model_in_keras_sequential_api_for_image_classification():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

"""
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
        .add(Conv2D(NUM_FILTERS[0], (KERNEL_HEIGHT, KERNEL_WIDTH), padding="valid",
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
        .add(BatchNormalization())

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
        .add(BatchNormalization())

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
        .add(BatchNormalization())

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
    # for a total of 100 Units (Weights and Biases)
    convolution_neural_network_tensorflow_keras_sequential_model\
        .add(Dense(128))

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
        .add(Dropout(0.25))

    # Softmax Classifier

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
"""

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

multi_classes_training_image_data_generator_for_preprocessing_with_data_augmentation = \
    image_data_generator_for_preprocessing_with_data_augmentation()

multi_classes_training_set_pokemon_data_augmentation_generator = \
    multi_classes_training_image_data_generator_for_preprocessing_with_data_augmentation\
    .flow(x=xs_features_training_set_pokemon, batch_size=BATCH_SIZE,
          y=ys_classes_training_set_pokemon, shuffle=False)

multi_classes_validation_image_data_generator_for_preprocessing_with_data_augmentation = \
    image_data_generator_for_preprocessing_with_data_augmentation()

multi_classes_validation_set_pokemon_data_augmentation_generator = \
    multi_classes_validation_image_data_generator_for_preprocessing_with_data_augmentation\
    .flow(x=xs_features_validation_set_pokemon, batch_size=BATCH_SIZE,
          y=ys_classes_validation_set_pokemon, shuffle=False)

# Initialise the Stochastic Gradient Descent (S.G.D.) Optimizer,
# with the Learning Rate of 5%, Momentum of 90% and Decay of (INITIAL_LEARNING_RATE / NUM_EPOCHS)
stochastic_gradient_descent_optimizer = SGD(learning_rate=INITIAL_LEARNING_RATE,
                                            momentum=0.9, decay=(INITIAL_LEARNING_RATE / NUM_EPOCHS))

# Create a Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification
cnn_model_in_keras_sequential_api_for_image_classification = \
    create_cnn_model_in_keras_sequential_api_for_image_classification()

# Compile the Model for the feed-forward Convolution Neural Network (C.N.N.),
# with the given Categorical Cross Entropy Loss/Error Function and
# the Stochastic Gradient Descent (S.G.D.) Optimizer
cnn_model_in_keras_sequential_api_for_image_classification.compile(loss="categorical_crossentropy",
                                                                   optimizer=stochastic_gradient_descent_optimizer,
                                                                   metrics=["accuracy"])

# Print the Log for the Fitting of the Model for the feed-forward Convolution Neural Network (C.N.N.)
print(f"\nFitting/Training the Model for the feed-forward Convolution Neural Network (C.N.N.) for {NUM_EPOCHS} Epochs "
      f"with a Batch Size of {BATCH_SIZE} and an Initial Learning Rate of {INITIAL_LEARNING_RATE}...\n")

# Train/Fit the Model for the feed-forward Convolution Neural Network (C.N.N.) for the given NUM_EPOCHS,
# with the Training Set for the Training Data and the Validation Set for the Validation Data
cnn_model_training_history = \
    cnn_model_in_keras_sequential_api_for_image_classification\
    .fit(multi_classes_training_set_pokemon_data_augmentation_generator.x,
         multi_classes_training_set_pokemon_data_augmentation_generator.y,
         steps_per_epoch=(NUM_EXAMPLES_FINAL_TRAINING_SET // BATCH_SIZE),
         epochs=NUM_EPOCHS,
         validation_data=(multi_classes_validation_set_pokemon_data_augmentation_generator.x,
                          multi_classes_validation_set_pokemon_data_augmentation_generator.y),
         validation_steps=(NUM_EXAMPLES_FINAL_VALIDATION_SET // BATCH_SIZE),
         batch_size=BATCH_SIZE)

# Print the final Log for the Fitting of the Model for the feed-forward Convolution Neural Network (C.N.N.)
print("\nThe Fitting/Training of the Model for the feed-forward Convolution Neural Network (C.N.N.) is complete!!!\n")
