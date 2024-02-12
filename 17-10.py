# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:42:35 2023

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
#Let us try to understand first hoe K means works for two
#dimeansional data
#For thst genearate random numbers in the range 0 to 1 
#and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#Create a empty dataframe with 0 rows and 2 columns
df_xy=pd.DataFrame(columns=["X","Y"])
#Assign the values of x and y to this columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind="scatter")
#KMeans(n_clusters=3) instance will be created
model1=KMeans(n_clusters=3).fit(df_xy)
'''
With data x and y ,apply Kmeans model
genrate scatter plot with scale/font=10
cmap=plt.cm.coolwarm:cool color combination'''


model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

univ1=pd.read_excel("C:/Data Science/Datasets/University_Clustering.xlsx")
univ1.describe()
univ=univ1.drop(["State"],axis=1) 
#We know that there is scale difference among the columns,which 
#We have either by using normalization or standrdization
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to the univ dataframe 
#For all the rows
df_norm=norm_fun(univ.iloc[:,1:])
'''
what will be the ideal cluster number,will it be 1,2,or 3
'''

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)#Total within sum of squares
'''Kmeans inertia also knowsn as sum of suqares errors
(or SSE),calcultes the sum of the distances of all points within 
cluster from the centriod of the point.It is the difference between
the observed values and the predicted value'''
TWSS
#As k value increses the twss value decreses
plt.plot(k,TWSS,'ro-');
plt.xlabel("No of clusters");
plt.ylabel("Total within the ss");
'''
How to select the values of k from elbow curve
when k changes from 2 to 3,then decrese in twss is higher than 
When k changes from 3 to 4
when k values changes from 5 to 6 decreses
in twss is considerably less,hence considered k=3
'''
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
univ['clust']=mb
univ.head()
univ=univ.iloc[:,[7,0,1,2,3,4,5,6]]
univ
univ.iloc[:,2:8].groupby(univ.clust).mean()
univ.to_csv("KMeans_university.csv",encoding="utf-8")
import os
os.getcwd()







#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel("C:/Data Science/Datasets/EastWestAirlines.xlsx")
df 
df.shape   
df.describe()
df.head()
