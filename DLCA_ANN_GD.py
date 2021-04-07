# Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from scipy import stats
import sympy as sp
from keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from keras import backend as K
from sklearn import preprocessing
from numpy import inf
import math
import random
import sklearn
import os


from GetPlots import GetErrorPlot, GetRegressPlot, GetPredErrorPlot, GetErrorDistPlot, GetInputsConvergencePlot
from DataImport import GetData, normData
from CustomFunction import inv_sigmoid
from ANN import build_model

################################################################################################################################
# Settings
################################################################################################################################
# Flag for saving and loading model
SaveModel = False
LoadModel = False


################################################################################################################################
# Train ANN
################################################################################################################################
# Load model
if LoadModel == True:
  
  model = keras.models.load_model('./ANNModel/3D_DLCA_ANN_Model.pb')

else:
    # Set custom inverse sigmoid activation function
    get_custom_objects().update({'inv_sigmoid': inv_sigmoid})

    # Set path and load data
    path = './Data/ML3DInput.csv'
    train_dataset, test_dataset, train_labels, test_labels, train_stats, dataset = GetData(path, 0.7)

    # Normalize data
    normed_train_data = normData(train_dataset,  train_stats['max'], train_stats['min'])
    normed_test_data = normData(test_dataset, train_stats['max'], train_stats['min'])

    # Build model
    shape = [len(train_dataset.keys())]
    model = build_model(shape)
    model.summary()

    # Train model
    eochs_ANN = 2000
    history = model.fit(normed_train_data, train_labels, epochs=eochs_ANN, validation_split=0.3, verbose=0,
                        callbacks=[tfdocs.modeling.EpochDots()])

    # MSE training plot
    GetErrorPlot(history, 'MSE', np.array([[0,2000], [0,0.0020]]), './Img')

    # MAE training plot
    GetErrorPlot(history, 'MAE', np.array([[0,2000], [0,0.06]]), './Img')

    # Make prediction
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=1)
    print("Testing set Mean Abs Error: {:5.2f} Fractal dimension".format(mae))
    test_predictions = model.predict(normed_test_data).flatten()

    # Plot regression
    GetRegressPlot(test_labels, test_predictions, './Img')

    # Plot error distribution
    GetPredErrorPlot(test_labels, test_predictions, './Img')

    # Save model
    if SaveModel == True:
    model.save('./ANNModel/3D_DLCA_ANN_Model.pb')


################################################################################################################################
# Inverse of the neural network
################################################################################################################################
from CustomFunction import tanh, tanhyperbolic, inv_sigmoid_inversion, sigmoid, linear
from InverseGradientDescent import forward, GetDelta, IGD
from ANN import ANN

# List activation functions of model
activation = [inv_sigmoid_inversion, tanhyperbolic, tanhyperbolic,  linear]

# Get weights and biases of the trained network
weights = model.get_weights()

# Create ANN object
ANNModel = ANN(model, activation, train_stats)

# Defining the trained network topology
nn = {}
radius = []
walkers = []
step = []

nn = ANNModel.nn

code_input_array = np.array([3, 0.06997, 0.5, 1.125])  # original inputs
code_input_norm = np.array(normData(code_input_array, train_stats['max'], train_stats['min']))  # normalisation of inpu
code_reshaped_input = code_input_norm.reshape(1, 4)
target_fractal_dimension = model.predict(code_reshaped_input)  # predicted Fractal dimension

# Initial inputs for gradient descent
ai_init = np.array([[-0.1, -0.1, -0.42, -.33]])

# Initialize lists
radius, conc, stepsS, stepsW = np.array([]), np.array([]), np.array([]), np.array([])
InputArray = [radius, conc, stepsS, stepsW]

# Define learning rate decay rate and epochs
learning_rate, decay_rate, epochs_inv = 0.01, 0, 20000

# Create inverse gradient descent object with all properties
IGDModel = IGD(learning_rate, decay_rate, epochs_inv, target_fractal_dimension, ai_init, code_input_norm)

# Initialize ai and neti
ai = IGDModel.ai_init
neti = IGDModel.neti_init

# Loop over all epochs
for i in range(epochs_inv):

    nn = forward(ai, activation, nn,  ANNModel.net_shape)  # calculate forward mapping
    out_fd = nn['A4']

    # Update neti by gradient descent
    neti = GetDelta( neti, ai, target_fractal_dimension, IGDModel.learning_rate, IGDModel.decay_rate, nn, IGDModel.code_input_norm)

    ai = sigmoid(neti)

    # Update input array
    for j in range(len(InputArray)):   
        InputArray[j] = np.append(InputArray[j], ai[0][j])
            
    if i % 1000 == 0:
        code_values = ((ai[0] + 0.5) * (train_stats['max'] - train_stats['min'])) + train_stats['min']
        print(code_values)
        print('Fractal Dimension:', out_fd)
        print('')

# Define input labels for plot
InputArray_label = ['radius', 'concentration', 'step size seeds', 'step size walkerss']

# Plot input convergence
GetInputsConvergencePlot(epochs_inv, InputArray, InputArray_label, 'unconstrained', './Img')


################################################################################################################################
# Calculate errors for random test data set
################################################################################################################################
# Get random test data set
random_test_data = test_dataset.sample(frac=0.154, random_state=1)
fd_labels = test_labels[random_test_data.index]
#fd_labels = random_test_data.pop('fractal dimension')

normalised_random = normData(random_test_data, train_stats['max'], train_stats['min'])

# Predicted Fractal dimension
predicted_test = model.predict(normalised_random)

# Initialize lists
radius, conc, stepsS, stepsW = np.array([]), np.array([]), np.array([]), np.array([])
InputArray = [radius, conc, stepsS, stepsW]

# Initialize list for errors
errors_unconstrained = []
errors_constrained = []
fd_constrained = []
fd_unconstrained = []

# Define learning rate decay rate and epochs
learning_rate, decay_rate, epochs_inv = 0.01, np.array([0, 0.1]), 13000

# Loop over constrained and unconstrained case with different decay rates
for rate in decay_rate:

    print("Running code for decay rate: ", rate)

    # Loop over all epochs
    for index, row in normalised_random.iterrows():

        # Determine input target
        input_array_new = np.array(row).reshape(1, 4)

        # Caluclate net i
        neti_input_target = inv_sigmoid_inversion(input_array_new)
        
        # Add random noise to ai
        randomiser = np.random.normal(0, 0.03)
        ai_init = input_array_new + randomiser

        # Determine fractal dimension as prediction by ANN
        target_fractal_dimension = model.predict(input_array_new)
        print('FD by ANN: ', target_fractal_dimension)

        # Create inverse gradient descent object with all properties
        IGDModel = IGD(learning_rate, rate, epochs_inv, target_fractal_dimension, ai_init, code_input_norm)

        # Initialize ai and neti
        ai = IGDModel.ai_init
        neti = IGDModel.neti_init

        for i in range(epochs_inv):

            nn = forward(ai, activation, nn,  ANNModel.net_shape)  # calculate forward mapping
            
            # Update neti by gradient descent
            neti = GetDelta(neti, ai, target_fractal_dimension, IGDModel.learning_rate, rate, nn, input_array_new)

            ai = sigmoid(neti)

            # Update input array
            for j in range(len(InputArray)):   
                InputArray[j] = np.append(InputArray[j], ai[0][j])

            code_values = ((ai[0] + 0.5) * (train_stats['max'] - train_stats['min'])) + train_stats['min']

            out_fd = nn['A4']

        print('FD by inversion: ', out_fd) 

        # Determine error for input array and for fractal dimension        
        lms = np.sqrt(np.sum((input_array_new - ai)**2))
        fd_error = np.sqrt(np.sum(out_fd - target_fractal_dimension)**2)

        # Assign errors either for constrained or unconstrained case
        if rate == 0:

            errors_unconstrained.append(lms)
            fd_unconstrained.append(fd_error)
        
        else:

            errors_constrained.append(lms)
            fd_constrained.append(fd_error)


# Plot error distribution for constrained and unconstrained case
GetErrorDistPlot(errors_unconstrained, errors_constrained, fd_unconstrained, fd_constrained, './Img')