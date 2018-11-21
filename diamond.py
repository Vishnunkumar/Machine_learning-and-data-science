import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.abspath(r'D:\Datas\diamond\diamonds.csv'))
print(data.isnull().sum() , data.dtypes)
print(data['cut'].value_counts(),data['clarity'].value_counts(),data['color'].value_counts())

#conversion to categorical
objects = data.select_dtypes(['object']).columns
data[objects] = data[objects].apply(lambda x:x.astype('category'))
cats = data.select_dtypes(['category']).columns
data[cats] = data[cats].apply(lambda x:x.cat.codes)

data.drop(['Unnamed: 0'],axis=1,inplace=True)
data.corr()>.6

#removing outliers
mean = data.mean()
std = data.std()
data_out = data[(data >= mean-(2*std)) & (data <= mean + (2*std))]
data_out.dropna(how='any',axis=0,inplace=True)

Price = data_out['price']
Feature = data_out.iloc[:,0:9]

from sklearn.cross_validation import train_test_split as tts

R_acc = []
for i in range(1,6):
    from sklearn.ensemble import RandomForestRegressor as RFR
    x_tr,x_te,y_tr,y_te = tts(Feature,Price,test_size=.2,random_state=255)
    RFR = RFR(max_features=i,n_estimators=i*(5),n_jobs=20)
    RFR.fit(x_tr,y_tr)
    R_acc.append(RFR.score(x_te,y_te))
    
A_acc = []
for i in range(1,6):
    from sklearn.ensemble import AdaBoostRegressor as RFR
    x_tr,x_te,y_tr,y_te = tts(Feature,Price,test_size=.2,random_state=255)
    RFR = RFR(n_estimators=i*(5),loss='linear')
    RFR.fit(x_tr,y_tr)
    A_acc.append(RFR.score(x_te,y_te))    

L_acc = []
for i in range(1,6):
    from sklearn.ensemble import GradientBoostingRegressor as RFR
    x_tr,x_te,y_tr,y_te = tts(Feature,Price,test_size=.2,random_state=255)
    RFR = RFR(n_estimators=i*(5),max_depth=i)
    RFR.fit(x_tr,y_tr)
    L_acc.append(RFR.score(x_te,y_te))
    
fig = plt.figure(figsize=(7,7))
plt.plot(np.arange(1,6),A_acc,c='r')
plt.plot(np.arange(1,6),R_acc,c='g')
plt.plot(np.arange(1,6),L_acc,c='b')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.title('RFR(g) vs ABR(r) vs Grad(b)')    