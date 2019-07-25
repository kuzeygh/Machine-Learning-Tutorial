# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:08:53 2019

@author: pc
"""
#random forest regressor r2_score

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:26:54 2019

@author: pc
"""

#evaluation of regression models
#residual=y-y_head
#here y_head is the result of real prediction results
#square_residual=residual^2
#so Squared Sum Residual( SSR ) is: sum( (y-y_head)^2 )
#Assume that we've found y_head_avg
#so Squared Sum Total is sum( (y-y_head_avg)^2 )
#r_squared=1-( SSR-SST ) ,the cloesest value of R2 is the better prediction results

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('C:/Users/pc/Desktop/udemy/python/decision.csv',sep=';',header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%%

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100, random_state=42 )
rf.fit(x,y)

print('7.8 level price: ',rf.predict([[7.8]]))

y_head=rf.predict(x)

plt.scatter(x,y,color='red' )
plt.plot(x,y_head,color='green')
plt.xlabel('tribun level')
plt.ylabel('price')
plt.show()
#%%
#evaluation of r_squared is
from sklearn.metrics import r2_score

print('r2_score: ',r2_score(y,y_head))
