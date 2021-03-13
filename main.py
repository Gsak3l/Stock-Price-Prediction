import matplotlib.pyplot as plt
import pandas_datareader as web
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense


def show_diagram(x_axis_tag, y_axis_tag, data_):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')
    plt.plot(data_)
    plt.xlabel(x_axis_tag, fontsize=18)
    plt.ylabel(y_axis_tag, fontsize=18)
    plt.show()


def optimize_dataframe():
    # keeping just the close column
    data = df.filter(['Close'])
    dataset_ = data.values
    training_data_len_ = math.ceil(len(dataset_) * 0.8)
    return training_data_len_, dataset_


def scaled_data(dataset_):
    scaler_ = MinMaxScaler(feature_range=(0, 1))
    scaled_data_ = scaler_.fit_transform(dataset_)
    return scaled_data_, scaler_


def create_training_dataset():
    train_data = scaled_data[0:training_data_len, :]
    x_train_ = []
    y_train_ = []

    for i in range(60, len(train_data)):
        x_train_.append(train_data[i - 60:i, 0])  # not including i
        y_train_.append(train_data[i, 0])

        if i <= 60:
            print(x_train_)
            print(y_train_)
            print()

    return x_train_, y_train_


def create_test_dataset():
    test_data = scaled_data[training_data_len - 60:, :]
    x_test_ = []
    y_test_ = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test_.append(test_data[i - 60: i, 0])

    x_test_ = np.array(x_test_)
    return x_test_, y_test_


def build_LSTM():
    model_ = keras.Sequential()
    model_.add(LSTM(50, return_sequences=True))
    model_.add(LSTM(50, return_sequences=True))
    model_.add(Dense(25))
    model_.add(Dense(1))

    model_.compile(optimizer='adam', loss='mean_squared_error')
    model_.fit(x_train, y_train, batch_size=1, epochs=1)

    return model_


def get_predictions_rmse():
    predictions_ = model.predict(x_test)
    predictions_ = scaler.inverse_transform(predictions_)  # unscaling the values

    # get the root mean squared error (RMSE)
    rmse_ = np.sqrt(np.mean(predictions_ - y_test))
    return predictions_, rmse_


if __name__ == '__main__':
    # getting the stock quote and visualizing some of the data
    df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-04-11')
    print(df.shape)

    show_diagram('Date', 'Close Price USD ($)', df['Close'])

    training_data_len, dataset = optimize_dataframe()

    scaled_data, scaler = scaled_data(dataset)

    x_train, y_train = create_training_dataset()
    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshaping the data because LSTM model expects 3 cols and not two
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = build_LSTM()

    # reshaping the data because LSTM model expects 3 cols and not two
    x_test, y_test = create_test_dataset()
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions, rmse = get_predictions_rmse()
