
# Import Python's Modules, Libraries and Packages

# Import the Sequential from the TensorFlow.Keras.Models Python's Module
from tensorflow.keras.models import Sequential

# Import the Convolution 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Conv2D

# Import the Activation Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Activation

# Import the Max Pooling 2D Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import MaxPooling2D

# Import the Flatten Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Flatten

# Import the Dense Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Dense

# Import the Dropout Layer from the TensorFlow.Keras.Layers Python's Module
from tensorflow.keras.layers import Dropout

# Import the Height of the Kernel of the Filters used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import KERNEL_HEIGHT

# Import the Width of the Kernel of the Filters used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import KERNEL_WIDTH

# Import the Height of the Pooling Matrix used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import POOLING_HEIGHT

# Import the Width of the Pooling Matrix used for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import POOLING_WIDTH

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

# Import the Number of Units of the last Dens Layer for the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_UNITS_LAST_DENSE_LAYER

# Import the Optimisers available to use for the the Model of
# the feed-forward Convolution Neural Network (C.N.N.)
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import AVAILABLE_OPTIMISERS_LIST

# Import the Number of Labels for the Datasets from the Pokemons' Data
# from the Parameters and Arguments Python's Custom Module
from project.tp1.libs.parameters_and_arguments import NUM_CLASSES_POKEMON_TYPES

KERNEL_HEIGHT = KERNEL_HEIGHT
KERNEL_WIDTH = KERNEL_WIDTH
POOLING_HEIGHT = POOLING_HEIGHT
POOLING_WIDTH = POOLING_WIDTH
STRIDE_HEIGHT = STRIDE_HEIGHT
STRIDE_WIDTH = STRIDE_WIDTH
NUM_UNITS_LAST_DENSE_LAYER = NUM_UNITS_LAST_DENSE_LAYER

# Function to create a Model for a feed-forward Convolution Neural Network (C.N.N.),
# for the Pokemons' Data, in Image Classification
def model_0_keras_sequential_api_for_image_classification(optimiser_id):

    # Create a Model for a feed-forward Convolution Neural Network (C.N.N.),
    # which is most appropriate for this type of problem (i.e., Image Classification),
    # using the Tensorflow Keras' Sequential API
    convolution_neural_network_tensorflow_keras_sequential_model = \
        Sequential(name='pokemon-images-multi-labels-classification')

    # --- 1st Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 1st Convolution 2D Layer, for the Input features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), using 32 Filters of a Kernel 3x3,
    # Same Padding and an Input Shape of (64 x 64 pixels), as also,
    # 3 Input Dimensions (for each Color Channel - RGB Color)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(32, (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding='same'))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # --- 2nd Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 2nd Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 64 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(64, (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding='same'))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a 3rd Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 64 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(128, (STRIDE_HEIGHT, STRIDE_WIDTH),
                    padding='same'))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # --- 3rd Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 4th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(128, (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding='same'))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a 5th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(128, (KERNEL_HEIGHT, KERNEL_WIDTH), padding='same'))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # --- 4th Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.) ---

    # Add a 6th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(256, (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding='same'))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a 7th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(256, (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding='same'))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a 8th Convolution 2D Layer, for the previous features of the Data/Images of
    # the Pokemons' Dataset given to the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), resulted from the previous layer,
    # using 128 Filters of a Kernel 3x3 and Same Padding
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Conv2D(256, (KERNEL_HEIGHT, KERNEL_WIDTH),
                    padding='same'))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Maximum Pooling 2D Sample-Based Discretization Process Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous layer of the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # with a 2x2 Pooling Size and 2x2 Stride Size
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(MaxPooling2D(pool_size=(POOLING_HEIGHT, POOLING_WIDTH),
                          strides=(STRIDE_HEIGHT, STRIDE_WIDTH)))

    # --- 5th Block of Layers for the Model for
    # the feed-forward Convolution Neural Network (C.N.N.),
    # for the Softmax Classifier, for the Multi-Label Problem ---

    # Add a Flatten Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.), with 512 Units
    # NOTE:
    # - This is needed to flatten the input into a single dimension for the features,
    #   which is what the next Dense Layer needs;
    convolution_neural_network_tensorflow_keras_sequential_model.add(Flatten())

    # Add a Dense Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # for a total of 512 Units (Weights and Biases)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Dense(NUM_UNITS_LAST_DENSE_LAYER))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Dense Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # for a total of 512 Units (Weights and Biases)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Dense(NUM_UNITS_LAST_DENSE_LAYER))

    # It is being used the ADAptive GRADient algorithm (ADA.GRAD.) Optimiser
    if optimiser_id == AVAILABLE_OPTIMISERS_LIST[3]:

        # Add a Dropout Layer of 50%, for the Regularization of Hyper-Parameters,
        # for the features of the Data/Images of the Pokemons resulted from
        # the previous Layer of the Model of the feed-forward
        # Convolution Neural Network (C.N.N.)
        convolution_neural_network_tensorflow_keras_sequential_model.add(Dropout(0.5))

    # Add a Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.)
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('relu'))

    # Add a Dense Layer to the features of the Data/Images of
    # the Pokemons resulted from the previous Layer of
    # the Model of the feed-forward Convolution Neural Network (C.N.N.),
    # for a total of 10 Units (Weights and Biases), for each type of Pokemon
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Dense(NUM_CLASSES_POKEMON_TYPES))

    # Add a Sigmoid as Activation Function Layer,
    # for the features of the Data/Images of the Pokemons resulted from
    # the previous Layer of the Model of the feed-forward
    # Convolution Neural Network (C.N.N.), for the Multi-Label Classifier
    convolution_neural_network_tensorflow_keras_sequential_model \
        .add(Activation('sigmoid'))

    # Return the Model for a feed-forward Convolution Neural Network (C.N.N.),
    # for the Pokemons' Data, in Image Classification, for the Multi-Label Problem
    return convolution_neural_network_tensorflow_keras_sequential_model