# Import packages
import numpy as np
import math

from DataImport import GetData, normData
from CustomFunction import tanh, tanhyperbolic, inv_sigmoid_inversion, sigmoid, linear, inv_sigmoid

# Extract layers of neural network
def forward(x, activation, nn, net_shape):
    nn['A0'] = x

    # Loop over network layers
    for i in range(len(net_shape) - 1):

        nn['Z' + str(i + 1)] = np.dot(nn['A' + str(i)], nn['weight' + str(i + 1)]) + nn['bias' + str(i + 1)]
        nn['A' + str(i + 1)] = activation[i](nn['Z' +str(i + 1)], derivative=False)

    return nn

# Get delta and update neti
def GetDelta( net_i, a_i, df_target, lrate, decay_rate, nn, code_target_values_norm):

    # Influence on error due to target
    Func1 = 2 * ((nn['A4']) - df_target)
    Func2 = ((np.dot(np.dot(nn['weight4'], tanh(nn['A3'], derivative=True)),
                nn['weight3']) @ tanh(nn['A2'], derivative=True).T).T @ nn['weight2'].T) * (sigmoid(nn['Z1'], derivative=True))

    # Influence on error regarding the input constraint 
    Func3 = 2 * (a_i - (code_target_values_norm)) * \
        (sigmoid(nn['Z1'], derivative=True))

    # Calculate delta
    delta = lrate * (Func1 @ Func2) + decay_rate * (Func3)

    # Update Delta net_i
    net_i = net_i - delta
    
    return net_i


# Define inverse gradient descent class for data management
class IGD:

    learning_rate = []
    decay_rate = []
    epochs_inv = []
    target_df = []
    ai_init = []
    neti_init = []
    code_input_norm = []

    # Initialize class
    def __init__(self, learning_rate, decay_rate, epochs_inv, target_df, ai_init, code_input_norm):

        # Assign learning rate
        self.learning_rate = learning_rate

        # Assign decay rate
        self.decay_rate = decay_rate

        # Assign number of epochs
        self.epochs_inv = epochs_inv

        # Assign target fractal dimension
        self.target_df = target_df

        # Assign initial ai
        self.ai_init = ai_init

        self.neti_init = inv_sigmoid(ai_init).numpy()

        # Assign normed code input 
        self.code_input_norm = code_input_norm