# Import packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.pyplot as plt

from CustomFunction import inv_sigmoid

# Build model
def build_model(shape):

    # Generate sequential model with 4 layers, first layer is not trainable and necessary for preventing gradient explosion
    model = keras.Sequential([layers.Dense(4, activation=inv_sigmoid, weights=[np.identity(4), np.zeros(4)], 
                                trainable=False, input_shape=shape),
                              layers.Dense(24,activation='tanh'),
                              layers.Dense(24,activation='tanh'),
                              layers.Dense(1)])

    # Assign Adam as optimizer
    optimizer = tf.keras.optimizers.Adam(0.01)

    # Compile model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


# Define ANN class for data management
class ANN:

    model = []
    weights = []
    activation = []
    nn={}
    net_shape = []
    data_stats = []

    # Initialize class
    def __init__(self, model, activation, data_stats):
        
        self.model = model

        #Get weights and biases of the trained network
        self.weights = model.get_weights()

        # List activation functions of model
        self.activation = activation
        
        # Get data information
        self.data_stats = data_stats

        # Get network shape and add initial input layer
        self.net_shape = [l.output_shape[1] for l in model.layers]
        self.net_shape.insert(0, 4)

        # Get weights and biases for each layer
        for j, i in enumerate(range(0, (len(self.net_shape)-1)*2, 2)):
            self.nn[('weight' + str(j+1))] = self.weights[i]
            self.nn[('bias' + str(j+1))] = self.weights[i+1]

