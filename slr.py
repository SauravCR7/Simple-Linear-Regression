# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:30:46 2017

@author: Saurav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Salary_Data.csv')

#creating dataset
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 1].values

#splitting into test and training sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=1/3,random_state=0)

"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
"""

#Simple Linear Regression on Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting test set results
y_pred=regressor.predict(x_test)

#Plotting Graphs [train]
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience[training set]')
plt.xlabel('salary')
plt.ylabel('experience')
plt.show

#Plotting Graphs [test]
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience[training set]')
plt.xlabel('salary')
plt.ylabel('experience')
plt.show





