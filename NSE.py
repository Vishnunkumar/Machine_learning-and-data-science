import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

r_d = os.path.abspath(r'E:\Placements\Datasets\NSE')
data = pd.read_csv(os.path.join(r_d,'banknifty.csv'))

#Preprocessing
print(data.dtypes)
print(data.isnull().sum())
print(data.shape)

#Conversion of datatypes of variables
data['time'] = pd.to_datetime(data['time'])
time = data['time'].dt.hour + (data['time'].dt.minute)*(10)**(-2) 
data['time'] = time

#Choosing features and Results
Result = data['close']
Features = data.iloc[:,1:6]

#Visualizing
data_r = data.iloc[:,1:6]
data_r.plot(figsize=(10,10),fontsize=10)
plt.title('NSE',fontsize=15)

#TIME SERIES ANALYSIS
#Visualizing
data_r['open'].plot(figsize=(6,6), linewidth=2)
data_r['open'].rolling(10000).mean().plot(figsize=(6,6), linewidth=2,c='g') 
data_r['close'].rolling(10000).mean().plot(figsize=(6,6), linewidth=2,c='g')
#increasing the window value smoothens the curve

#Seasonal patterns in time series dataset
#using first order differencing to remove the trend and only have the seasonility
data_r['open'].diff().plot(figsize=(10,10), linewidth=2,c='g')


















