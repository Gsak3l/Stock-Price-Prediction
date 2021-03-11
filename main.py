import matplotlib.pyplot as plt
import pandas_datareader as web
import math
from sklearn.preprocessing import MinMaxScaler


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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_ = scaler.fit_transform(dataset_)
    return scaled_data_


if __name__ == '__main__':
    # getting the stock quote and visualizing some of the data
    df = web.DataReader('GME', data_source='yahoo', start='2012-01-01', end='2021-04-11')
    print(df.shape)

    show_diagram('Date', 'Close Price USD ($)', df['Close'])

    training_data_len, dataset = optimize_dataframe()

    scaled_data = scaled_data(dataset)
    print(scaled_data)
