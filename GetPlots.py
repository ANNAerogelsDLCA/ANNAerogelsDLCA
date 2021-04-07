import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import numpy as np


# Plot MAE or MSE
def GetErrorPlot(history, err, lims, path):

    # Create plt
    plt.figure(figsize=(5 ,5),dpi=500)
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=10)
    plotter.plot({err : history}, metric = err.lower() , smoothing_std = 10)
    
    # Assign limits
    plt.xlim(lims[0,:])
    plt.ylim(lims[1,:])

    # Assign labels
    plt.ylabel(err, fontsize=14)
    plt.xlabel('Epochs', fontsize=14)

    # Save figure
    plt.savefig(path + '/Plot_' + err + '.eps', format='eps', bbox_inches = 'tight')

# Plot regression
def GetRegressPlot(true_x, pred_x, path):

    # Plot regression with scatter
    plt.figure(figsize=(5 ,5),dpi=300)
    plt.scatter(true_x, pred_x , marker= '.', color = 'red')

    # Assign labels
    plt.xlabel('True Fractal Dimension', fontsize=14)
    plt.ylabel('Predicted Fractal Dimension', fontsize=14)
    
    # Limit range of fractal dimension between 2 and 3
    lim = [2, 3]
    plt.xlim(lim)
    plt.ylim(lim)

    # Plot regression line with pred_x = true_x
    _ = plt.plot(lim, lim)

    # Save figure
    plt.savefig(path + '/Plot_Regression.eps' , format = 'eps', bbox_inches = 'tight')


# Plot error distribution
def GetPredErrorPlot(true_x, pred_x, path):

    # Calculate error
    error = pred_x- true_x

    # Create figure
    plt.figure(figsize=(5 ,5),dpi=300)

    # Create histrogram
    plt.hist(error, bins = 40)
    plt.xlabel("Prediction Error ",  fontsize=14)
    _ = plt.ylabel("Count",  fontsize=14)

    # Print R2 score
    r2 = r2_score(true_x, pred_x)
    print('R2 accuracy is',r2)

    # Save figure
    plt.savefig(path + '/Plot_Error_Distribution.eps' , format = 'eps')


# Plot error distribution
def GetErrorDistPlot(errors_unconstrained, errors_constrained, fd_unconstrained, fd_constrained, path):

    # Create figure for input errors
    plt.figure(figsize=(5 ,5),dpi=300)

    logbins = np.logspace(-10,1,30)
    logbins2 = np.logspace(-10,1,30)

    plt.hist(errors_unconstrained, bins=logbins,label= 'Unconstrained inversion',alpha = 0.5)
    plt.hist(errors_constrained, bins=logbins2, label= 'Constrained inversion',alpha = 0.5)

    plt.xscale('log')
    plt.legend(prop={'size': 9}, loc= 'upper right',bbox_to_anchor=(0.5, 1))

    plt.xlabel("Predicted inputs error (log scale)  ",  fontsize=14)
    plt.ylabel("Count",  fontsize=14)
    plt.savefig(path + '/Error_inputs.pdf',bbox_inches = "tight")

    # Create figure for fd errors
    plt.figure(figsize=(5 ,5),dpi=300)

    logbins = np.logspace(-9,1,45)
    logbins2 = np.logspace(-9,1,45)

    plt.hist(fd_unconstrained, bins=logbins,label= 'unconstraint inversion',alpha = 0.7)
    plt.hist(fd_constrained, bins=logbins2, label= 'constraint inversion',alpha = 0.5)

    plt.xscale('log')
    plt.legend(prop={'size': 9}, loc= 'upper right',bbox_to_anchor=(1, 1))

    plt.xlabel("Predicted fractal dimension error (log scale)  ",  fontsize=14)
    plt.ylabel("Count",  fontsize=14)
    plt.savefig(path + '/Error_fractal_dimension.pdf' ,bbox_inches = "tight")




# Plot convergence of inputs 
def GetInputsConvergencePlot(epochs, data, data_label, constrainType, path):

    x = np.linspace(0,epochs, epochs)
    
    # Create figure
    plt.figure(figsize=(5 ,5), dpi =300)

    # Assign title and labels
    plt.title('Gradient Plot', fontsize=12)
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel(' Normalised predicted input values', fontsize=12)
    
    # Loop over inputs 
    for j in range(len(data)):  
        plt.plot(x, data[j],label= data_label[j])

    # Add legends
    plt.legend(prop={'size': 9}, loc= 'upper right',bbox_to_anchor=(1, 0.9))
    
    # Save figure
    plt.savefig(path + '/Plot_InputConvergence_' + constrainType + '.eps', format= 'eps' ,bbox_inches = "tight")
