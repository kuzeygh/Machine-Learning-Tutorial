# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:29:56 2019

@author: pc
"""

#Decision Tree Regression

#%%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataFrame=pd.read_csv('C:/Users/pc/Desktop/udemy/python/decision.csv',sep=';',header=None)
#x is considered as tribun level
#y is considered as price

#all rows and column 0
x=dataFrame.iloc[:,0].values.reshape(-1,1)
#all rows and column 1

y=dataFrame.iloc[:,1].values.reshape(-1,1)

tree_regression_model=DecisionTreeRegressor()
tree_regression_model.fit(x,y)

#for example let us predict the tribun level 5.5's price
print( tree_regression_model.predict([[5.5]]))
plt.scatter(x,y,color='red')
#in scatter x and y must be the same size so first scatter and then change x's arange
#x=np.arange(min(x),max(x),0.01).reshape(-1,1) #uncomment this row and see better results

y_head=tree_regression_model.predict(x)


plt.plot(x,y_head,color='green')
plt.xlabel('tribun level')
plt.ylabel('price')
plt.show()
#the x values which are in the same interval 
#gets the same value of y,but since we have less data it doesn't appear like that
#so lets make some arrangements of x data



