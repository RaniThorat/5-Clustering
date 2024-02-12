# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:39:34 2023

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
auto=pd.read_csv("C:/Data Science/Datasets/AutoInsurance.csv.xls")
auto1=auto.drop(["Policy"],axis=1)
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_fun(auto.iloc[:,1:])


b=df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Clustring dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance");
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

