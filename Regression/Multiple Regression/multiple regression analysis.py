# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:28:10 2017

@author: Preeti
"""

# control shift enter
# Data pre processing
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('50_Startups.csv')

#format
# go to variable explorer. click on dataset. click on format. change .3g to .0f (float)

#gets created of type object bcause the variables are of different type. type X in console
# to view data

x = dataset.iloc[:, :-1].values
# iloc[rows,columns] function : means all. all rows and all columns except last

y = dataset.iloc[:, 4].values
#all rows and only 3rd column (which is purchased in our case)


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#at this point x will become float64 from type object


#avoiding dummy variable trap
x = x[:,1:]

#splitting dataset into training and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)


#fit multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting test set results
y_pred  = regressor.predict(x_test)


print(regressor.coef_)




# Plot outputs
plt.scatter(x_train, y_train,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()