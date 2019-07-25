# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
# Implementations Of Machine Learning

#By this point, you should have Scikit-Learn already installed. 
#If not, get it, along with Pandas and matplotlib!

#%%
#Sİmple Linear Regression

#Y ≈ β 0 + β 1 X.

#this is our equation of simple linear regression model, here b0 is called 
#bias or intercept and b1 is coefficient

#Fİrst we'll import the required libraries 
#matplotlib for data visualization
#pandas for dataframe,list,series data structures of python
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

#after creating the equation now we find b0 and b1
#there are two ways of finding the bo,bo is the point where graph intersects
#the yaxis.So,give 0 to x.First way of finding b0 is,

b0=linear_reg_model.predict([[0]])
print('bo: ',b0)
#we must use 0 in 2D type array,cause we can predict more than one values
#Second way of finding b0 is using intercept_ method
b0=linear_reg_model.intercept_ 
print('b0: ',b0)
#these two result must be same

#and lets find b1 by using coefficient_ method
b1=linear_reg_model.coef_
print('b1: ',b1)

#and now we can use our simple AI,we'll show you two ways
#first manually predict 'How much money does a worker that has 30 years of
#experience gain ?'

experience_=30

new_income=b0+b1*experience_
print('new income: ', new_income )

#another way to use it -automatically- is,
print( 'new income 2:',linear_reg_model.predict([[experience_]]))


#%%
#it is time to see that already fitted line,let's plot it
#but first we must predict all the x values in our graph
#we'll call that prediction results as y_head
import numpy as np

array=np.arange(min(x),max(x),0.01) #fix the (1500,)
array=array.reshape(-1,1)

y_head=linear_reg_model.predict(array)
plt.scatter( x , y )
plt.plot( array , y_head , color='red')
plt.show()

