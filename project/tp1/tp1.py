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

# Import Python's Modules, Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import AVAILABLE_OPTIMISERS_LIST

# Import the Number of Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_AVAILABLE_OPTIMISERS

# Import the function to execute the Model of Multi-Class Classification
# for all the Available Optimisers, from the Module for
# the Multi-Class Classification Problem in the Project
from pokemon_images_multi_class_classification import \
    execute_model_of_multi_class_classification_for_all_available_optimisers

# Import the function to execute the Model of Multi-Label Classification
# for all the Available Optimisers, from the Module for
# the Multi-Label Classification Problem in the Project
from pokemon_images_multi_label_classification import \
    execute_model_of_multi_label_classification_for_all_available_optimisers

# Import the function to execute the Model of Multi-Label Classification
# for all the Available Optimisers, from the Module for
# the Image Masking/Semantic Segmentation Problem in the Project
from pokemon_images_semantic_segmentation import \
    execute_model_of_semantic_segmentation_for_all_available_optimisers

# Import the function to execute the Pre-Trained MobileNet Model
# of Multi-Class Classification for all the Available Optimisers,
# from the Module for the ImageNet Classification Problem in the Project
from pokemon_image_net_classification import \
    execute_mobile_net_model_multi_class_classification_for_all_available_optimisers

# Import the function to execute the Pre-Trained MobileNet Model
# of Multi-Label Classification for all the Available Optimisers,
# from the Module for the ImageNet Classification Problem in the Project
from pokemon_image_net_classification import \
    execute_mobile_net_model_multi_label_classification_for_all_available_optimisers

"""
# Print the initial separator
print('------------------------------------------------------------------------------------')

# Print the logging for the execution of
# the Model of the Multi-Class Classification for all the Available Optimisers
print('\n')
print('Executing the Pokemon Multi-Class Classification, for all the Available Optimisers...')

# Execute the Model of the Multi-Class Classification for all the Available Optimisers
optimisers_training_loss_means_multi_class, \
    optimisers_training_accuracy_means_multi_class, \
    optimisers_validation_loss_means_multi_class, \
    optimisers_validation_accuracy_means_multi_class, \
    optimisers_true_testing_loss_means_multi_class, \
    optimisers_true_testing_accuracy_means_multi_class = \
    execute_model_of_multi_class_classification_for_all_available_optimisers()

# Print a separator, for the logging for the execution of
# the Model of the Multi-Class Classification for all the Available Optimisers
print('------------------------------------------------------------------------------------')

# Print the logging for the execution of
# the Model of the Multi-Label Classification for all the Available Optimisers
print('\n')
print('Executing the Pokemon Multi-Label Classification, for all the Available Optimisers...')

# Execute the Model of the Multi-Label Classification for all the Available Optimisers
optimisers_training_loss_means_multi_label, \
    optimisers_training_accuracy_means_multi_label, \
    optimisers_validation_loss_means_multi_label, \
    optimisers_validation_accuracy_means_multi_label, \
    optimisers_true_testing_loss_means_multi_label, \
    optimisers_true_testing_accuracy_means_multi_label = \
    execute_model_of_multi_label_classification_for_all_available_optimisers()

# Print a separator, for the logging for the execution of
# the Model of the Multi-Label Classification for all the Available Optimisers
print('------------------------------------------------------------------------------------')

# Print the logging for the execution of
# the Model of the Image Masking/Semantic Segmentation for all the Available Optimisers
print('\n')
print('Executing the Pokemon Image Masking/Semantic Segmentation, for all the Available Optimisers...')

# Execute the Model of the Semantic Segmentation for all the Available Optimisers
optimisers_training_loss_means_semantic_segmentation, \
    optimisers_training_accuracy_means_semantic_segmentation, \
    optimisers_validation_loss_means_semantic_segmentation, \
    optimisers_validation_accuracy_means_semantic_segmentation, \
    optimisers_true_testing_loss_means_semantic_segmentation, \
    optimisers_true_testing_accuracy_means_semantic_segmentation = \
    execute_model_of_semantic_segmentation_for_all_available_optimisers()

# Print a separator, for the logging for the execution of
# the Model of the Image Masking/Semantic Segmentation for all the Available Optimisers
print('------------------------------------------------------------------------------------')
"""

# Print the logging for the execution of
# the Model of the Multi-Class Classification, with MobileNet Pre-Trained Model,
# using ImageNet Weights, for all the Available Optimisers
print('\n')
print('Executing the Pokemon Multi-Class Classification, with MobileNet Pre-Trained Model, '
      'using ImageNet Weights, for all the Available Optimisers...')

# Execute the Model of the Multi-Class Classification,
# with MobileNet Pre-Trained Model, using ImageNet Weights,
# for all the Available Optimisers
optimisers_training_loss_means_mobile_net_multi_class, \
    optimisers_training_accuracy_means_mobile_net_multi_class, \
    optimisers_validation_loss_means_mobile_net_multi_class, \
    optimisers_validation_accuracy_means_mobile_net_multi_class, \
    optimisers_true_testing_loss_means_mobile_net_multi_class, \
    optimisers_true_testing_accuracy_means_mobile_net_multi_class = \
    execute_mobile_net_model_multi_class_classification_for_all_available_optimisers()

# Print a separator, for the logging for the execution of
# the Model of the Multi-Class Classification, with MobileNet Pre-Trained Model,
# # using ImageNet Weights, for all the Available Optimisers
print('------------------------------------------------------------------------------------')


"""
# Print the logging for the execution of
# the Model of the Multi-Label Classification, with MobileNet Pre-Trained Model,
# using ImageNet Weights, for all the Available Optimisers
print('\n')
print('Executing the Pokemon Multi-Label Classification, with MobileNet Pre-Trained Model, '
      'using ImageNet Weights, for all the Available Optimisers...')

# Execute the Model of the Multi-Label Classification,
# with MobileNet Pre-Trained Model, using ImageNet Weights,
# for all the Available Optimisers
optimisers_training_loss_means_mobile_net_multi_label, \
    optimisers_training_accuracy_means_mobile_net_multi_label, \
    optimisers_validation_loss_means_mobile_net_multi_label, \
    optimisers_validation_accuracy_means_mobile_net_multi_label, \
    optimisers_true_testing_loss_means_mobile_net_multi_label, \
    optimisers_true_testing_accuracy_means_mobile_net_multi_label = \
    execute_mobile_net_model_multi_label_classification_for_all_available_optimisers()

# Print a separator, for the logging for the execution of
# the Model of the Multi-Label Classification, with MobileNet Pre-Trained Model,
# # using ImageNet Weights, for all the Available Optimisers
print('------------------------------------------------------------------------------------')

"""

"""
# Print the logging for the execution of
# the Model of the Image Masking/Semantic Segmentation for all the Available Optimisers
print('\n\n')
print('-------- Final Results for the all the Problems and Available Optimisers --------')

# Print the logging for the results of the Multi-Class Classification Problem
print('\n')
print('---- Results of the Multi-Class Classification ----')

# For each Optimiser available
for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

    # Print the respective Means (Averages) for the Losses and Accuracies
    # of the predictions made by the current Optimiser on all the Sets
    print(' - %s: [ train_loss = %.12f ; train_acc = %.12f |'
          ' val_loss = %.12f ; val_acc = %.12f |'
          ' test_loss = %.12f ; test_acc = %.12f ]'
          % (AVAILABLE_OPTIMISERS_LIST[num_optimiser],
             optimisers_training_loss_means_multi_class[num_optimiser],
             optimisers_training_accuracy_means_multi_class[num_optimiser],
             optimisers_validation_loss_means_multi_class[num_optimiser],
             optimisers_validation_accuracy_means_multi_class[num_optimiser],
             optimisers_true_testing_loss_means_multi_class[num_optimiser],
             optimisers_true_testing_accuracy_means_multi_class[num_optimiser]))


# Print the logging for the results of the Multi-Label Classification Problem
print('\n')
print('---- Results of the Multi-Label Classification ----')

# For each Optimiser available
for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

    # Print the respective Means (Averages) for the Losses and Accuracies
    # of the predictions made by the current Optimiser on all the Sets
    print(' - %s: [ train_loss = %.12f ; train_acc = %.12f |'
          ' val_loss = %.12f ; val_acc = %.12f |'
          ' test_loss = %.12f ; test_acc = %.12f ]'
          % (AVAILABLE_OPTIMISERS_LIST[num_optimiser],
             optimisers_training_loss_means_multi_label[num_optimiser],
             optimisers_training_accuracy_means_multi_label[num_optimiser],
             optimisers_validation_loss_means_multi_label[num_optimiser],
             optimisers_validation_accuracy_means_multi_label[num_optimiser],
             optimisers_true_testing_loss_means_multi_label[num_optimiser],
             optimisers_true_testing_accuracy_means_multi_label[num_optimiser]))


# Print the logging for the results of the Image Masking/Semantic Segmentation Problem
print('\n')
print('---- Results of the Image Masking/Semantic Segmentation ----')

# For each Optimiser available
for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

    # Print the respective Means (Averages) for the Losses and Accuracies
    # of the predictions made by the current Optimiser on all the Sets
    print(' - %s: [ train_loss = %.12f ; train_acc = %.12f |'
          ' val_loss = %.12f ; val_acc = %.12f |'
          ' test_loss = %.12f ; test_acc = %.12f ]'
          % (AVAILABLE_OPTIMISERS_LIST[num_optimiser],
             optimisers_training_loss_means_semantic_segmentation[num_optimiser],
             optimisers_training_accuracy_means_semantic_segmentation[num_optimiser],
             optimisers_validation_loss_means_semantic_segmentation[num_optimiser],
             optimisers_validation_accuracy_means_semantic_segmentation[num_optimiser],
             optimisers_true_testing_loss_means_semantic_segmentation[num_optimiser],
             optimisers_true_testing_accuracy_means_semantic_segmentation[num_optimiser]))

"""

# Print the logging for the results of the Multi-Class Classification Problem, with ImageNet Weights
print('\n')
print('---- Results of the Multi-Class Classification, with the ImageNet Weights ----')

# For each Optimiser available
for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

    # Print the respective Means (Averages) for the Losses and Accuracies
    # of the predictions made by the current Optimiser on all the Sets
    print(' - %s: [ train_loss = %.12f ; train_acc = %.12f |'
          ' val_loss = %.12f ; val_acc = %.12f |'
          ' test_loss = %.12f ; test_acc = %.12f ]'
          % (AVAILABLE_OPTIMISERS_LIST[num_optimiser],
             optimisers_training_loss_means_mobile_net_multi_class[num_optimiser],
             optimisers_training_accuracy_means_mobile_net_multi_class[num_optimiser],
             optimisers_validation_loss_means_mobile_net_multi_class[num_optimiser],
             optimisers_validation_accuracy_means_mobile_net_multi_class[num_optimiser],
             optimisers_true_testing_loss_means_mobile_net_multi_class[num_optimiser],
             optimisers_true_testing_accuracy_means_mobile_net_multi_class[num_optimiser]))

"""

# Print the logging for the results of the Multi-Label Classification Problem, with ImageNet Weights
print('\n')
print('---- Results of the Multi-Label Classification, with the ImageNet Weights ----')

# For each Optimiser available
for num_optimiser in range(NUM_AVAILABLE_OPTIMISERS):

    # Print the respective Means (Averages) for the Losses and Accuracies
    # of the predictions made by the current Optimiser on all the Sets
    print(' - %s: [ train_loss = %.12f ; train_acc = %.12f |'
          ' val_loss = %.12f ; val_acc = %.12f |'
          ' test_loss = %.12f ; test_acc = %.12f ]'
          % (AVAILABLE_OPTIMISERS_LIST[num_optimiser],
             optimisers_training_loss_means_mobile_net_multi_label[num_optimiser],
             optimisers_training_accuracy_means_mobile_net_multi_label[num_optimiser],
             optimisers_validation_loss_means_mobile_net_multi_label[num_optimiser],
             optimisers_validation_accuracy_means_mobile_net_multi_label[num_optimiser],
             optimisers_true_testing_loss_means_mobile_net_multi_label[num_optimiser],
             optimisers_true_testing_accuracy_means_mobile_net_multi_label[num_optimiser]))
"""