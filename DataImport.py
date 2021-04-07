# Import packages
import pandas as pd

# Normalize dataset
def normData(x, x_max, x_min):
    return (x - x_min)/(x_max-x_min)-0.5

# Read data from path and split into test and train data sets
def GetData(path, split_ratios):

  # Read data
  df= pd.read_csv(path)
  df= df.drop(columns =["Unnamed: 0"])

  # Prepare training and test data
  train_data = df.sample(frac= split_ratios,random_state=1)
  test_data = df.drop(train_data.index)
  test_data = test_data.sample(frac=1, random_state=1)

  # Get information for data
  data_stats = train_data.describe()

  # Remove labels to be predicted
  data_stats.pop("fractal dimension")
  data_stats = data_stats.transpose()

  # Assign physically realistic minimum and maximum values for all features
  data_stats['min']['radius'] = 0
  data_stats['max']['radius'] = 10
  data_stats['min']['conc'] = 0
  data_stats['max']['conc'] = 0.2
  data_stats['min']['steps walkers'] = 0
  data_stats['max']['steps walkers'] = 10
  data_stats['min']['steps seeds'] = 0
  data_stats['max']['steps seeds'] = 10
  
  # Extract labels for train and test data
  train_labels = train_data.pop('fractal dimension')
  test_labels = test_data.pop('fractal dimension')

  return [train_data, test_data, train_labels, test_labels, data_stats, df]
