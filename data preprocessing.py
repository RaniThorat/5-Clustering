# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:49:16 2023

@author: Admin
"""

import pandas as pd
import matplotlib.pylab as plt
#Now import file from data set and create a dataframe
univ1=pd.read_excel("C:/Data Science/Datasets/University_Clustering.xlsx")
#We have one col "State" which really not useful we will drop it
univ=univ1.drop(["State"],axis=1)
#We know that there is scale difference among the col,
#Which we have to remove
#Either by using normalization or standrdization
#Whenever there is mixed data apply normalization
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now applay this normalization function to univ dataframe
#for all the rows and col from 1 untill end
#Since 0th col has university name hence skipped
df_norm=norm_fun(univ.iloc[:,1:])
#You can check the df_norm dataframe which is scaled
#You can applay 



b=df_norm.describe()
#Before you apply clustering you need to plot dendogram first
#Now to create dendrogram,we need to measure distance,
#We have to import linkage 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#Linkage  function gives us hierarchical or aglomerative clustering
#ref the help for linkage 
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Clustring dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance");
#ref help of dendrogram
#sch,dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendrogram()
#applying agglomerative clustering choosing 3 as cluster
#From dendrogram
#Whatever has been displayed in the dendrogram it is not clustering
#It is just showing number of possible clustering
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#Apply lalbels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign THis series to univ dataframe as col and name the col as "Clustring"
univ['clust']=cluster_labels
#We want to relocate the col 7 to 0th position
univ1=univ.iloc[:,[7,1,2,3,4,5,6]]
#Now check the univ1 dataframe
univ1.iloc[:,2:].groupby(univ1.clust).mean()
#From the output cluster 2 has got highest top10
#lowest accept ratio,best faculty ratio and highest expenses
#highest graduestes  ratio
univ1.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()
