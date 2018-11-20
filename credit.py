import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.abspath(r'D:\Datas\credic\C_C.csv'))
print(data.dtypes,data.isnull().sum())

#Checking for outliers
mean = data.mean()
std = data.std()
data_in = data[(data >= mean-2*(std)) & (data <= mean+2*(std))]
data_in.dropna(axis=0,inplace=True)

#Removing multicollinearity
data['PAY_F'] = data['PAY_AMT1'] + data['PAY_AMT2'] + data['PAY_AMT3'] + data['PAY_AMT4'] + data['PAY_AMT5'] + data['PAY_AMT6']
data['BILL_F'] = data['BILL_AMT1'] + data['BILL_AMT2'] + data['BILL_AMT3'] + data['BILL_AMT4'] + data['BILL_AMT5'] + data['BILL_AMT6']
drop_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','BILL_AMT1',
             'BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
data.drop(drop_cols,axis=1,inplace=True)

Default = data['Default']
data.drop(['Default'],axis=1,inplace=True)
plt.scatter(data['PAY_0'],data['PAY_2'])
#Removing multicollinear variables
data['PAY_5'] = (data['PAY_5'] + data['PAY_6'])/2
data['PAY_2'] = (data['PAY_2'] + data['PAY_3'])/2
multi = ['PAY_6','PAY_4','PAY_3']
data.drop(multi,axis=1,inplace=True)
data.drop(['PAY_2'],axis=1,inplace=True)

#Varying n_estimators in RFC and GBC to analyze the variance in accuracy
Accuracy = []
for i in range(1,5):
    from sklearn.ensemble import RandomForestClassifier as RFC
    RFC = RFC(max_features=3,n_jobs=5,n_estimators=i*(10))
    RFC.fit(x_tr,y_tr)
    Accuracy.append(RFC.score(x_te,y_te))

accuracy = []
for i in range(1,5):
    from sklearn.ensemble import GradientBoostingClassifier as RFC
    RFC = RFC(max_leaf_nodes=5,n_estimators=i*(10))
    RFC.fit(x_tr,y_tr)
    accuracy.append(RFC.score(x_te,y_te))
    
fig = plt.figure(figsize=(7,7))
plt.plot(np.arange(1,5),Accuracy,c='r')
plt.plot(np.arange(1,5),accuracy,c='g')
plt.xlabel('variable')
plt.ylabel('accuracy')
plt.title('RFC(red) vs GBC(green)')
