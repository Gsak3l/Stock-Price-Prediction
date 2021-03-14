import matplotlib.pyplot as plt
import pandas_datareader as web
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense


def show_diagram(title, x_axis_tag, y_axis_tag, data_):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.plot(data_)
    plt.xlabel(x_axis_tag, fontsize=18)
    plt.ylabel(y_axis_tag, fontsize=18)
    plt.show()


def optimize_dataframe():
    # keeping just the close column
    data_ = df.filter(['Close'])
    dataset_ = data_.values
    training_data_len_ = math.ceil(len(dataset_) * 0.8)
    return training_data_len_, dataset_, data_


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
    predictions_ = scaler.inverse_transform(predictions_)

    # getting the root mean squared error RMSE)
    rmse_ = np.sqrt(np.mean(predictions_ - y_test) ** 2)
    return predictions_, rmse_


def plot_data():
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # visualize the data, my function show_diagram() is completely useless
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


if __name__ == '__main__':
    # getting the stock quote and visualizing some of the data
    df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-04-11')

    show_diagram('Close Price History', 'Date', 'Close Price USD ($)', df['Close'])

    training_data_len, dataset, data = optimize_dataframe()

    scaled_data, scaler = scaled_data(dataset)

    x_train, y_train = create_training_dataset()
    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshaping the data because LSTM model expects 3 cols and not two
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = build_LSTM()

    x_test, y_test = create_test_dataset()
    # reshaping the data because LSTM model expects 3 cols and not two
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions, rmse = get_predictions_rmse()

    plot_data()

    apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-04-11')
    new_df = apple_quote.filter((['Close']))

    # get the last 60 day closing price values and convert the dataframe to an array
    last_60 = new_df[-60].values
    last_60_scaled = scaler.transform(last_60)

    # appending the last 60 days
    x_test_2 = [last_60_scaled]
    x_test_2 = np.array(x_test_2)
    x_test_2 = np.reshape(x_test_2, (x_test.shape[0], x_test_2[1], 1))

    # predicted prices
    predicted_price = model.predict(x_test_2)
    predicted_price = scaler.inverse_transform(predicted_price)
    print(predicted_price)

""""
Description: This program uses an artificial recurrent network called Long Short Term Memory (LSTM)
to predict the closing stock price of a corporation (in this case Apple Inc.) using the past 60 day stock price
as input data
"""
