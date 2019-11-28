#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
LSTM Model for predicting (average) stock price using fundamental and
technical data.

Fundamental and technical data for a given company is used to predict future
average stock price. By future average, it is meant that n future days of
price will be predicted and the average of the predicted prices will be used
to give the overall estimate.

An LSTM model with walk-forward validation is used.
'''

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#from keras.layers import LSTM
#from keras.layers import RepeatVector
#from keras.layers import TimeDistributed

from math import sqrt, ceil
import numpy as np
import pandas as pd

from matplotlib import pyplot
import tensorflow as tf
#from tensorflow.keras import models
'''from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
'''
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
LSTM = tf.keras.layers.LSTM
RepeatVector = tf.keras.layers.RepeatVector
TimeDistributed = tf.keras.layers.TimeDistributed


# TODO: CHECK I AM GRABBING THE RIGHT COLUMN for my model output.
# TODO: Check I am inputting things correctly to each model.
# TODO: Make sure I am inputting correct datasets for each part of model traning.
# TODO: Train on a few more epochs.
# TODO: Check future result outputs.
# TODO: Get final predicted price.
# TODO: Clean up comments.

def minMaxScaler(pd_col):
    '''
    Create minMaxScaler. minMaxScaler chosen so the relative spacing of
    features are maintained.

    Args:
        pd_col: Pandas Dataframe column.

    Returns:
        Scaled column.
    '''
    c_max = pd_col.max()
    c_min = pd_col.min()

    # If no range found set everything to zero.
    # This generally happens for columns that have all zero entries.
    if c_max - c_min == 0:
        return pd_col*0

    return (pd_col - c_min)/(c_max-c_min)


def rescaleLabel(prices, c_max, c_min):
    '''
    Takes the scaled predicted prices and rescales them back to their original
    scale. This is important because the output should be the share price on
    the basis of the company being evaluated, not some scaled version of share
    price.

    Args:
        prices: Array of scaled predicted prices (values between 0-1 because
        minMaxScaler is used).
        c_max = max value of column before scaling.
        c_min = min value of column before scaling.

    Return:
        Rescale (or rather unscaled) share price.
    '''
    return prices*(c_max-c_min) + c_min


def clean_up_input_data(df_data):
    fund_cols = ['Gross_Dividends___Common_Stock',
      'Net_Income_Before_Taxes', 'Normalized_Income_Avail_to_Cmn_Shareholders',
      'Operating_Expenses', 'EBIT', 'Total_Assets__Reported',
      'Total_Debt', 'Total_Equity', 'Total_Liabilities', 'Total_Long_Term_Debt']


    tech_cols = ['Date','Volume']

    label_cols = ['Price_Close']


    all_cols = tech_cols + fund_cols + label_cols


    df_model_data = df_data.loc[:,all_cols]

    # Drop any Null data (should already by done by cleaning)
    df_model_data = df_model_data.dropna()

    # Sort by date
    df_model_data = df_model_data.sort_values(by='Date').reset_index(drop=True)

    # Set date as index (it is not used as feature in training, but rather the
    # sequence of inputs is what is important).
    df_model_data = df_model_data.set_index('Date')

    # Make sure all columns are converted to numeric form.
    df_model_data = df_model_data.apply(pd.to_numeric)

    # Scale columns.
    df_model_data = df_model_data.apply(minMaxScaler)

    return df_model_data


'''
Model looks at daily data for past 3 years and predicts next 6 months.
Loss function is the RMSE for day 1 to day 180 (6 months ahead).
'''


def split_to_train_val_test_sets(pd_data, window=180):
    '''
    Take orignal dataset and split it up.
    pd_data.shape = (x,y)

    output data shape = (x/window, window, y)
    Args:
        pd_data: Pandas dataframe of original data.
        window: Number of time steps for each sample.

    Return:
        train, validation, and test datasets, all of which are numpy arrays
        and the date ranges associated with each of the datasets.
    '''
    # Window for LSTM is 180 days.
    #window = 180

    days_in_data = pd_data.shape[0]

    num_perfect_splits = int(days_in_data/window)

    # Up to 65% of data is used for training.
    training_splits = int(0.65*num_perfect_splits)

    # Half of remaining data is for testing.
    test_splits = int(ceil((num_perfect_splits - training_splits)*0.5))

    # Remaining data is for validation.
    val_splits = num_perfect_splits - training_splits - test_splits

    train = pd_data.iloc[:(training_splits*window),:]
    test = pd_data.iloc[(training_splits*window):\
                        (training_splits+test_splits)*window,:]
    val = pd_data.iloc[(training_splits+test_splits)*window:\
                       (num_perfect_splits*window),:]
    
    # Get dates associated with each dataset
    train_dates = np.array(train.index)
    test_dates = np.array(test.index)
    val_dates = np.array(val.index)
    

    # Convert to numpy arrays
    train = train.to_numpy()
    test = test.to_numpy()
    val = val.to_numpy()

    # Restructure data into 180 day windows
    train = np.array(np.split(train, training_splits))

    test = np.array(np.split(test, test_splits))

    val = np.array(np.split(val, val_splits))

    return (train, test, val),(train_dates,test_dates,val_dates)


def get_daily_increment_windows(train,n_input=180,n_out=180):
    '''
    Make daily increment windows.
    Data is orignially sliced as follows:
        [day1, day2, ... day 180]
        [day 181, day 182, ... day 360]
        ...
    with all features associated with each day.
    To be able to see how each possible window predicts the next window, we
    reorganize the data as follows:
        [day1, day2, ... day 180]
        [day2, day3, ... day 181]
        ... 178 more of these
        [day 180, day 181, ... day 359]
        [day 181, day 182, ... day 360]
        ...

    Args:
        train: Numpy array of training data.
        n_input: Size of input window used to make predictions.
        n_out: Size of output window that is being predicted.

    Returns:
        Numpy array of data reorganized (X), and numpy array of corresponding
        actual prices for the output window (y).
    '''

    # Flatten data back so that windows are combined into one contiguous
    # matrix.
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # Step over the entire history one time step (day) at a time.
    for i in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        # Ensure we have enough data for next instance.
        if out_end <= len(data):
            X.append(data[in_start:in_end,:])
            # Price_Close (the label), is the last column in the array.
            y.append(data[in_end:out_end,-1])
        # Move one time step ahead.
        in_start +=1
    return np.array(X), np.array(y)



def make_forecast(model, history, n_input=180):
    '''
    Make forcast on the most recent history size n_input, to predict next time
    steps.

    Args:
        model: fitted LSTM model.
        history: historical data.
        n_input: Size of window to use to make next prediction.

    Return:
        Predicted values, yhat.
    '''
    # Flatten data.
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # Retrieve last observations for input data.
    input_x = data[-n_input:, :]
    # Reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # Forecast the next n_output (default is 180 days).
    yhat = model.predict(input_x, verbose=2)
    # We only want the vector forecast.
    #yhat = yhat[0]

    # TODO: RESCALE PRICE BACK TO ORIGINAL SCALE.
    # Could also try not scaling price and seeing how things work.
    return yhat[0]


def evaluate_forecasts(actual, predicted):
    '''
    Evaluate forecasts. Each array has shape (x,y) where x is the number
    of timesteps predicts are made for and y is the number of days prediction
    are made.
    Ex. actual.shape = (10,100)
    Predictions were made for 10 timesteps, and each prediction was for 100
    different days.

    Args:
        actual: Numpy array of actual results (price on given day).
        predicted: Numpy array of predicted results.

    Return:
        Overall RMSE and list of RMSE for each timestep
    '''
    # calculate an RMSE score for each day
    mse_for_all_days = np.sqrt(np.square(actual-predicted)).reshape(
            (predicted.shape[0]*predicted.shape[1]))
    #np.sqrt(np.square(actual-predicted).mean(axis=0))
    # Calculate overall RMSE
    score = np.sum(np.sqrt(np.square(actual-predicted)))
    return score, mse_for_all_days


def calc_average_pct_error(actual, pred):
    '''
    Calculate the average percent error of predicted results.
    
    Args:
        actual: Actual prices.
        pred: Predicte prices.
    
    Return:
        Average percent error
    '''
    return (np.abs((pred-actual))/actual).mean()


def build_model(train, n_input):
    # Prepare data.
    train_x, train_y = get_daily_increment_windows(train, n_input)
    # Define parameters.
    verbose, epochs, batch_size = 1, 5, 32
    n_timesteps = train_x.shape[1]
    n_features = train_x.shape[2]
    n_outputs = train_y.shape[1]
    # Reshape output into [samples, timesteps, features].
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # Define model.
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps,
                                                        n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # Fit network.
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
              verbose=verbose)
    return model


def evaluate_model(model, train, test, n_input, max_price, min_price):
    # Fit model
    #model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = make_forecast(model, history, n_input)
        # store the predictions
        yhat_sequence = rescaleLabel(yhat_sequence, max_price, min_price)
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    predictions = predictions.reshape((predictions.shape[0],
                                       predictions.shape[1]))
    actual_prices = rescaleLabel(test[:, :, -1], max_price, min_price)
    score, scores = evaluate_forecasts(actual_prices, predictions)
    return score, scores, predictions, actual_prices


def predict_future_results(model, train, n_input, max_price, min_price):
    # history is a list of weekly data
    history = [x for x in train]
    # predict the week
    yhat_sequence = make_forecast(model, history, n_input)
    # store the predictions
    yhat_sequence = rescaleLabel(yhat_sequence, max_price, min_price)
    return yhat_sequence

def summary_of_results(dataset_name ,score, pred, actual):
    '''
    Print out summary statistics of model predictions.
    
    Args:
        dataset_name: train, test, or val.
        score: sum of RMSE for all predictions.
        pred: predictions.
        actual: actual values.
    '''
    avg_pct_err = calc_average_pct_error(pred, actual)

    print('Summary of {} predictions'.format(dataset_name))
    print('---------------------------')
    print('RMSE: {} \t\t'.format(score))
    print('Average percent error: {} \t\t'.format(avg_pct_err))
    print('---------------------------')
    print('\n\n')

df_data = pd.read_csv('/Users/paindox/Documents/Sixth Year/SENG 550/TMP/facebook_tech_and_fund_data.csv')
max_price = df_data[['Price_Close']].max()[0]
min_price = df_data[['Price_Close']].min()[0]

df_model_data = clean_up_input_data(df_data)


(train, test, val), (train_dates,test_dates,val_dates) = \
  split_to_train_val_test_sets(df_model_data)

# Set the window size for prediction (and by default input to lstm as well).
window = 180

# Build model on training data
print('Building model on training data')
training_model = build_model(train, window)

# Get model performance on training dataset
test_score, test_scores, test_pred, test_actual = evaluate_model(
        model,train, test, window, max_price, min_price)

# Get summary of results on test set.
summary_of_results('test', test_score, test_pred, test_actual)

# Build model on training and test data (to predict validation data)
print('Retraining model on training and test sets combined')
train_test = np.concatenate((train,test))
validation_model = build_model(train_test, window)

# Get model performance on validation dataset
val_score, val_scores, val_pred, val_actual = evaluate_model(
        validation_model, train_test, val, window, max_price, min_price)
summary_of_results('validation', val_score, val_pred, val_actual)

# Make future predictions, using validation set.
print('Building model on full dataset to make final predictions')
full_dataset = np.concatenate((train,test,val))
prediction_model = build_model(full_dataset, window)
pred_results = predict_future_results(prediction_model,full_dataset,n_input,
                                      max_price, min_price)




# Combine results with the date they are for


# TODO: Save the model weights somewhere.