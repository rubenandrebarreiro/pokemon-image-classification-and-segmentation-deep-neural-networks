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

# Import the Auxiliary Function to Load the Data for the Project, from the TP1_Utils' Module
from tp1_utils import load_data


# Constants

# The Width of the Images/Pictures
IMAGES_WIDTH = 64

# The Height of the Images/Pictures
IMAGES_HEIGHT = 64

# The Number of Channels for RGB Colors
NUM_CHANNELS_RGB = 3

# The Number of Channels for Black and White Colors
NUM_CHANNELS_BW = 1


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


