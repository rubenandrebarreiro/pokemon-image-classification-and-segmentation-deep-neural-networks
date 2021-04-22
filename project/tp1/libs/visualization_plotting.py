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

Visualization/Potting Module for the the Project

"""

# Import Python's Modules, Libraries and Packages

# Import PyPlot from the Matplotlib Python's Library
from matplotlib import pyplot as py_plot

# Import the Number of Epochs
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_EPOCHS

# Import the List of Available Optimisers
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import AVAILABLE_OPTIMISERS_LIST

# Import the Number of Available Optimisers
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_AVAILABLE_OPTIMISERS

# Import the Learning Rates for the Available Optimisers
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import INITIAL_LEARNING_RATES

# Import the Matplotlib Colors of the Plotting for the Available Optimisers
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import OPTIMISERS_COLORS_MATPLOTLIB


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
