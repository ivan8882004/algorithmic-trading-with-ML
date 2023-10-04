import yfinance as yf
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os


def get_data(stock, interval, delay):
    # Get historical price data form yfinance
    df = yf.download(stock)

    # Take the Closing Price column only
    df = df[["Close"]]

    # Take the amount of records we need for prediction
    df = df[-(interval+delay):]
    dataset = df.values

    return dataset


def prediction_set_building(interval, scaled_data):

    x_prediction = []

    for i in range(interval, len(scaled_data)):
        # Append each of the num_of_interval of records as an element.
        x_prediction.append(scaled_data[i-interval:i, 0])

    return x_prediction


def predict_data(stock, interval, delay):

    dataset = get_data("AAPL", interval, delay)

    # Scale the target closing price
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_evaluation = prediction_set_building(interval, scaled_data)

    # Transform into Array type and then reshape for Keras Model
    x_evaluation = np.array(x_evaluation)
    i,j = x_evaluation.shape
    x_evaluation = np.reshape(x_evaluation,(i,j,1))      

    # Build the path of the place to load the model
    root_dir = os.getcwd()
    path = os.path.join(root_dir, "Resources/"+stock+"_model")
    loaded_model = load_model(path)

    # Do the prediction
    predictions = loaded_model.predict(x_evaluation)

    # Inverse transform the outcome
    predictions = scaler.inverse_transform(predictions)

    return predictions

