# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:25:50 2019

@author: pc
"""
#r2 score linear regression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

#Second we'll import the csv files we need,to do that:
#  1.Choose file explorer and make sure that your csv file is in the correct directory
#  2.Use the read_csv command to import the required csv file

#  or choose variable explorer and click on the import data icon,find the file
#  that you want to import and click next,choose the data type dataframe and done.
#  if you import it as array then you should make some arrangement after that 
#  if we import the dataset as array,skiprows=1.
#  then convert it by using df=pd.DataFrame( nameofvariable(array),columns=['col1','col2'])

dataFrame=pd.read_csv('C:/Users/pc/Desktop/udemy/python/job2.csv')

#after getting this csv file visualize this data by using scatter plot

plt.scatter( dataFrame.experience,dataFrame.income )
plt.xlabel('Experince')
plt.ylabel('Income')
plt.show()
#here we saw a normal scatter plot here x is a feature that affects the values of y
#experience is a feature that affects the income

#%%

#now we'll go on with learning how to fit  line into this graph
#we'll be working with sklearn library,so import it

from sklearn.linear_model import LinearRegression

#initialize the linear regression model
linear_reg_model=LinearRegression()

#here why don't we use just x=dataFrame.experince ,cause this is a series
#or why don't we use x=dataFrame.experince.values,let's show the size of it

x=dataFrame.experience.values.shape
print(x)
 #see it gives (14,) that means 14 rows and 1 column but it is
#not preferable by sklearn library so reshape it to (14,1) by using
#reshape(-1,1) method as below
x=dataFrame.experience.values.reshape(-1,1)
y=dataFrame.income.values.reshape(-1,1 )
#print( type(x)) x is now an array has the shape (14,1)

#fitting the line into our scatter plot

linear_reg_model.fit( x,y )
y_head=linear_reg_model.predict( x )

print('r_square score: ',r2_score(y,y_head))