# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:37:27 2023

@author: Admin
"""
import pandas as pd
import numpy as np
univ1=pd.read_excel("C:/Data Science/Datasets/University_Clustering.xlsx")
univ1.describe()

univ1.describe()
uni=univ1.drop(["State"],axis=1)

from sklearn.decomposition  import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Considering only numerical data
uni.data=uni.iloc[:,1:]

#Normalizing the numerical data
uni_normal=scale(uni.data)
uni_normal

pca=PCA(n_components=6)
pca_values=pca.fit_transform(uni_normal)


#THe amount of variance that each pca explain is
var=pca.explained_variance_ratio_
var

#PCA weights
#pca.components_
#pca.components_[0]

#Cumultative variance
var1=np.cumsum(np.round(var,decimals=4)*100)

#Variance plot for PCA components obtained
plt.plot(var1,color="red")

#PCA score
pca_values
pca_data=pd.DataFrame(pca_values)
pca_data.columns="comp0","comp1","comp3","comp4","comp5","comp6"
final=pd.concat([uni.univ1,pca_data[:,0:3]],axis=1)

#This is uinv col of uni data frame
#Scatter diagram

import matplotlib.pyplot  as plt
ax=final.plot(x="comp0",y="comp1",kind="scatter",figsize=(12,8))
final[['comp0','comp1','univ']].apply(lambda x:ax.text(*x),axis=1)