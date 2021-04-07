# Import packages
import numpy as np
from numpy import inf
from keras import backend as K
import math

# Tangens hyperbolicus function
def tanh(x, derivative=False):
    if (derivative):

        return (1 - (x**2))

    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))


# Tangens hyperbolicus function with inf check
def tanhyperbolic(x, derivative=False):
    output = ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    # Check if output contains any NaN values
    if np.isnan(output).any():

        if -math.inf in x:
            output = np.nan_to_num(output, nan=-1)

        if -math.inf in x:

            output = np.nan_to_num(output, nan=1)

    if (derivative):

        return (1 - (x**2))

    return output

# Inverted symmetric sigmoid function with inf check
def inv_sigmoid_inversion(x, derivative=False):

    if np.isnan(x).any() or np.isinf(x).any():
        x = np.nan_to_num(x, neginf=-0.5, posinf=0.5)

    if (derivative):

        if np.isnan(x).any() or np.isinf(x).any():
            x = np.nan_to_num(x, neginf=-0.5, posinf=0.5)

        return -1 / ((x**2) - 0.25)

    return (-K.log((1 / (x + 0.5) - 1)))


# Inverted symmetric sigmoid function
def inv_sigmoid(x, derivative=False):

    if (derivative):
        return -1 / ((x**2) - 0.25)

    return (-K.log((1 / (x + 0.5) - 1)))


# Symmetric sigmoid function 
def sigmoid(x, derivative=False):

    if (derivative):
        return np.exp(-x) / ((np.exp(-x) + 1)**2)

    return ((1 / (1 + np.exp(-x))) - 0.5)


# Linear Function
def linear(x, derivative=False):
    return x
