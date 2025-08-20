# End-to-end example for SNN Toolbox with VGG11
#
# You can get this CNN's weights (and others) from this Colab notebook:
# https://colab.research.google.com/drive/1-3Tz-SLTFZqXswdtQMOIY_kjpF3tNnn6?usp=sharing
#
# Currently this does not work with most pretrained models due to the old TensorFlow version
# required by SNN Toolbox , the suggested workflow is to instantiate and train your own.

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from typing import Optional

import networkx as nx
import traceback
import signal
import code
import sys
import re

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

from keras_to_graph import *

in_interactive_mode = False

def signal_handler(signal, frame) -> None:
    global in_interactive_mode
    if in_interactive_mode:
        print('EXITING...')
        sys.exit(0)
    else:
        print('\nHANDLING TERMINATION...\n')
        stack = traceback.format_stack(frame)
        print('------------ stack -----------')
        print(''.join(stack[:-1])[:-1])
        print('------------------------------')
        #stop_threads()
        time.sleep(0.2)
        print('\nTERMINATION RECEIVED - SWITCHING TO INTERACTIVE MODE\n[type "exit()" or press "ctrl+c" again to terminate the program]\n')
        in_interactive_mode = True
        code.interact(local = globals())
        in_interactive_mode = False

signal.signal(signal.SIGINT, signal_handler)

try:
    # Configuration
    subset_size = 100
    path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp', str(time.time())))
    os.makedirs(path_wd, exist_ok = True)

    # Load and preprocess CIFAR-10
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize pixel values
    x_test = x_test / 255.0
    # Convert class vectors to binary class matrices (one-hot encoding)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    x_norm = x_test[:min(500, len(x_test))]  # Use first 500 or all if fewer available

    # Save dataset
    print("Saving processed data...")
    np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
    np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
    np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_norm)

    del x_test, y_test, x_norm, _

    # LOAD AND PREPARE MODEL #
    # Expected neurons: 194.058
    def vgg11_cifar10(input_shape = (32, 32, 3), num_classes = 10):
        inputs = tf.keras.Input(shape = input_shape)

        # Block 1
        x = layers.Conv2D(64, (3, 3), padding = "same", use_bias = False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3), padding = "same", use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3), padding = "same", use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(256, (3, 3), padding = "same", use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3), padding = "same", use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(512, (3, 3), padding = "same", use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3), padding = "same", use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(512, (3, 3), padding = "same", use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Classifier
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation = "relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation = "relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation = "softmax")(x)

        return Model(inputs, outputs, name = "VGG11_CIFAR10")

    print("Loading .h5 model...")
    model_name = 'vgg11_cifar10_model'
    model = vgg11_cifar10()
    # TODO: Set this to the file you downloaded from the above Colab notebook!
    load_weights_compat(model, f'{model_name}.weights.npz')

    # Add compilation info for SNN Toolbox compatibility
    model.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

    # Save model
    model.summary()
    model.save(os.path.join(path_wd, model_name + '.h5'))
    extract_neuron_graph(model, use_layer_type_as_name = True, directly_to_file = os.path.join(path_wd, model_name + '.graphml'), compact = True)

    del model

    # SNN TOOLBOX CONFIGURATION #
    #############################
    configparser = import_configparser()
    config = configparser.ConfigParser()

    config['paths'] = {
        'path_wd': path_wd,
        'dataset_path': path_wd,
        'filename_ann': model_name
    }

    config['tools'] = {
        'evaluate_ann': True,
        'normalize': True,
        'scale_weights': True
    }

    config['simulation'] = {
        'simulator': 'INI',
        'duration': 100,        # Increased for complex model
        'num_to_test': 50,      # Test 100 samples
        'batch_size': 50,       # MUST COINCIDE WITH THE ABOVE!
        'dt': 0.1
    }

    config['input'] = {
        'poisson_input': False
    }

    config['output'] = {
        'plot_vars': {},
        'log_vars': {'spiketrains_n_b_l_t'} # Generates the log used for spike frequency calculation
    }

    # Store config file
    config_filepath = os.path.join(path_wd, 'config')
    with open(config_filepath, 'w') as configfile:
        config.write(configfile)

    # RUN SNN TOOLBOX #
    ###################
    main(config_filepath)

except:
    print(traceback.format_exc())
    in_interactive_mode = True
    code.interact(local = globals())
    in_interactive_mode = False