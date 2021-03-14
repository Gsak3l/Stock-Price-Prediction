import yahoo_fin.stock_info as si
import pandas as pd
import json
import matplotlib.pyplot as plt

plt.style.use('seaborn')

tsla = si.get_quote_table('TSLA')
print(tsla)

df = pd.read_csv('TSLA.csv')
df = df.set_index(pd.DatetimeIndex(df['Date'].values))
df.pop('Date')
print(df.head())

plt.figure(figsize=(12.2, 4.5))
plt.title('Close Price', fontsize=18)
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
