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

# Import the function to execute the Model of Multi-Class Classification
# for all the Available Optimisers, from the Module for
# the Multi-Class Classification Problem in the Project
from pokemon_images_multi_class_classification import \
    execute_model_of_multi_class_classification_for_all_available_optimisers

# Import the function to execute the Model of Multi-Label Classification
# for all the Available Optimisers, from the Module for
# the Multi-Class Classification Problem in the Project
from pokemon_images_multi_label_classification import \
    execute_model_of_multi_label_classification_for_all_available_optimisers


# Execute the Model of Multi-Class Classification for all the Available Optimisers
execute_model_of_multi_class_classification_for_all_available_optimisers()

# Execute the Model of Multi-Label Classification for all the Available Optimisers
execute_model_of_multi_label_classification_for_all_available_optimisers()
