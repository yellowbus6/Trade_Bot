import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

def request_data(ticker):
    data = requests.get(
        f'''https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/2023-07-01/2023-07-07?adjusted=true&sort=asc&limit=50000&apiKey=c18qNVBiro5YcLi5dpyiTVapE1v50m9q''')
    data = data.json()
    data = data['results']
    data = pd.DataFrame(data)
    data = data.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    data.to_csv('data1.csv')
    return data

def keltner_channel(df, window=20, multiplier=1.5):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    middle_line = tp.rolling(window=window).mean()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    upper_band = middle_line + (multiplier * atr)
    lower_band = middle_line - (multiplier * atr)
    df['kelt_high'] = upper_band
    df['kelt_mid'] = middle_line
    df['kelt_low'] = lower_band
    return df

def bollinger_bands(df, window=20, num_sd=2):
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()
    df['bb_mid'] = rolling_mean
    df['bb_high'] = rolling_mean + (rolling_std * num_sd)
    df['bb_low'] = rolling_mean - (rolling_std * num_sd)
    return df




df = pd.read_csv('data1.csv')


#calc bb/keltner
keltner_data = keltner_channel(df)
bb_data = bollinger_bands(df)


#Plot keltner
plt.plot(keltner_data['kelt_high'], color='red')
#plt.plot(keltner_data['kelt_mid'])
plt.plot(keltner_data['kelt_low'], color='red')


#Plot BB Bands
plt.plot(bb_data['bb_high'], color='black')
plt.plot(bb_data['bb_low'], color='black')


#Plot Price
plt.plot(df['Close'], color='blue')


squeeze_points = (df['bb_high'] < df['kelt_high']) & (df['bb_low'] > df['kelt_low'])


plt.scatter(df.index[squeeze_points], df['Close'][squeeze_points], color='green', label='Squeeze Points')


def calculate_cmo(data, period=14):
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['diff'] = data['TP'].diff()
    data['pos_diff'] = data['diff'].apply(lambda x: max(0, x))
    data['neg_diff'] = data['diff'].apply(lambda x: max(0, -x))
    data['pos_avg'] = data['pos_diff'].rolling(window=period).sum()
    data['neg_avg'] = data['neg_diff'].rolling(window=period).sum()
    data['CMO'] = ((data['pos_avg'] - data['neg_avg']) /
    (data['pos_avg'] + data['neg_avg'])) * 100
    return data


def plot_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.scatter(data.index[data['CMO'] > 0], data['Close'][data['CMO'] > 0],
    color='blue', label='CMO Cross', marker='o')
    plt.scatter(data.index[data['CMO'] < - 0], data['Close'][data['CMO'] < -0],
    color='red', label='CMO Cross ', marker='o')
    plt.title('Stock Price with CMO Crosses')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)


if __name__ == "__main__":
    ticker = 'TSLA'
    data = request_data(ticker)
    data = keltner_channel(data)
    data = bollinger_bands(data)

    # Calculate CMO
    period = 14
    data = calculate_cmo(data, period)

    # Plot data
    plot_data(data)
    plt.show()
