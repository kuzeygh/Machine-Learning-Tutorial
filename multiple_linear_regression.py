# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:43:09 2019

@author: pc
"""
#%%
#Multiple Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataFrame=pd.read_csv('C:/Users/pc/Desktop/udemy/python/job2.csv')

#our equation will be like
#y=b0+b1*x1+b2*x2+...+bn*xn

x=dataFrame.iloc[:,[0,1]].values
#we'll be separating dataFrame as we're taking all rows and just the 0. and 1. columns
#why we are taking just 0. and 1. columns? Cause They are the predictors...
print(type(x))
print(x.shape) #Check the shape,it is usable

y=dataFrame.income.values.reshape(-1,1)

#initializing our multiple linear regression model
multiple_linear_regression_model=LinearRegression()
multiple_linear_regression_model.fit(x,y)

#Let's find the intercept and coefficients...

print( 'b0: ',multiple_linear_regression_model.intercept_ )
print( 'b1: ',multiple_linear_regression_model.coef_ )#we'll see two values at there

#now let's predict that how much many does a man that has
# 5 years of experience and is 35 years old and another man that
#has 10 years of experience and 35 years old gain money ?
multiple_linear_regression_model.predict([[35,5],[35,10]])


y_head=multiple_linear_regression_model.predict(x)

#both age and experience affects the income,plot both of them in scatter mode
plt.scatter( dataFrame.experience,dataFrame.income )
plt.scatter( dataFrame.age,dataFrame.income )

plt.xlabel('Experince')
plt.ylabel('Income-Age')

#draw the fitting line
plt.plot( x , y_head , color='red')
plt.show()



