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

Pre-Processing Module for the Project

"""

# Import Python's Modules, Libraries and Packages

# Import the Auxiliary Function to Load the Data for the Project,
# from the TP1_Utils' Python's Module
from project.tp1.tp1_utils import load_data

# The Number of Examples (Images/Masks) for the final Training Set
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_EXAMPLES_FINAL_TRAINING_SET

# Import the Image's Data Generator from
# the Tensorflow.Keras.PreProcessing.Image Python's Module
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

    print(ys_labels_testing_set.shape[1])

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
        fill_mode='nearest')

    # Return the Images' Data Generator for Pre-Processing with Data Augmentation
    return image_data_generator
