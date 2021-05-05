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

Images' Set Generator Module for the Project

"""

# Import Python's Modules, Libraries and Packages

# Import the Images to Pic Function
# from the TP1_Utils' Python's Module
from tp1_utils import images_to_pic


# Function to generate Images' Set, for the several sets
# (i.e., Training, Validation and Testing Sets)
def generate_images_sets(xs_features_training_set_pokemon, xs_masks_training_set_pokemon,
                         xs_features_training_augmented_set_pokemon, xs_masks_training_augmented_set_pokemon,
                         xs_features_validation_set_pokemon, xs_masks_validation_set_pokemon,
                         xs_features_validation_augmented_set_pokemon, xs_masks_validation_augmented_set_pokemon,
                         xs_features_testing_set_pokemon, xs_masks_testing_set_pokemon):

    # Print the informative log about the generation and saving of
    # the Images of Pokemons and Masks for the Training Set
    print('\nGenerating the Images of Pokemons and Masks for the Training Set...')

    # Generate the Figure with the Images, from the xs (features) of the Training Set
    images_to_pic('pokemons-training-set-images.png',
                  xs_features_training_set_pokemon)

    # Generate the Figure with the Images, from the xs (masks) of the Training Set
    images_to_pic('pokemons-masks-training-set-images.png',
                  xs_masks_training_set_pokemon)

    # Generate the Figure with the Images, from the xs (features) of the Training Augmented Set
    images_to_pic('pokemons-training-augmented-set-images.png',
                  xs_features_training_augmented_set_pokemon)

    # Generate the Figure with the Images, from the xs (masks) of the Training Augmented Set
    images_to_pic('pokemons-masks-training-augmented-set-images.png',
                  xs_masks_training_augmented_set_pokemon)

    # Print the informative log about the generation and saving of
    # the Images of Pokemons and Masks for the Training Set
    print('The generated Images of Pokemons and Masks for the Training Set saved!!!\n')

    # Print the informative log about the generation and saving of
    # the Images of Pokemons and Masks for the Validation Set
    print('\nGenerating the Images of Pokemons and Masks for the Validation Set...')

    # Generate the Figure with the Images, from the xs (features) of the Validation Set
    images_to_pic('pokemons-validation-set-images.png',
                  xs_features_validation_set_pokemon)

    # Generate the Figure with the Images, from the xs (masks) of the Validation Set
    images_to_pic('pokemons-masks-validation-set-images.png',
                  xs_masks_validation_set_pokemon)

    # Generate the Figure with the Images, from the xs (features) of the Validation Augmented Set
    images_to_pic('pokemons-validation-augmented-set-images.png',
                  xs_features_validation_augmented_set_pokemon)

    # Generate the Figure with the Images, from the xs (masks) of the Validation Augmented Set
    images_to_pic('pokemons-masks-validation-augmented-set-images.png',
                  xs_masks_validation_augmented_set_pokemon)

    # Print the informative log about the generation and saving of
    # the Images of Pokemons and Masks for the Validation Set
    print('The generated Images of Pokemons and Masks for the Validation Set saved!!!\n')

    # Print the informative log about the generation and saving of
    # the Images of Pokemons and Masks for the Testing Set
    print('\nGenerating the Images of Pokemons and Masks for the Testing Set...')

    # Generate the Figure with the Images, from the xs (features) of the Testing Set
    images_to_pic('pokemons-testing-set-images.png',
                  xs_features_testing_set_pokemon)

    # Generate the Figure with the Images, from the xs (masks) of the Testing Set
    images_to_pic('pokemons-masks-testing-set-images.png',
                  xs_masks_testing_set_pokemon)

    # Print the informative log about the generation and saving of
    # the Images of Pokemons and Masks for the Testing Set
    print('The generated Images of Pokemons and Masks for the Testing Set saved!!!\n')
