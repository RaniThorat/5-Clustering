# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:44:33 2023

@author: Admin
"""

from mlxtend.frequent_patterns import apriori,association_rules
#Here we are going to usee transactional data where in size of each row is not
#we can not use pandas to load this unstructured data
#Here function called open() is uesd 
#Create an empty list
gro=[]
with open("D:/Data Science/Datasets/groceries.csv.xls") as f:gro=f.read()
#Splitting the data into the seprate trnasactions using
#Sepertor ,it is comma and
#We can use new line charachter "\n"
gro=gro.split("\n")
#Earlier groceries data structure was in the string format
#Now it will change to the binary data structure
#9836,each item is comma seprated
#our main aim is to calculate #A,#C
#we will have to seprate out each item from each transaction
gro_list=[]
for i in gro:
    gro_list.append(i.split(","))
#split function will seperate each item from each list ,whereever
#in order to generate association rules,
#you can directly use gro_list
#now let us seperate out each item from gro_list
all_gro_list=[ i for item in gro_list for i in item]
#you will get alll the items occured in all transaction
#we will get 43368 items in various transactions


#Now let us count the frequency of each items
#we will import collections package which has counter
#function which will convert 
from collections import Counter
item_fre=Counter(all_gro_list)
#item_fre is basically dic having x[0] as key and x[1]=values
#we want to access values and sort based on the count that occured in it.
#it will show the count of each item purchesded in every transactions
#Now let us sort these frequencied in asc order
item_fre=sorted(item_fre.items(),key=lambda x:x[1])
#When we execute this,item fre will be in sorted form 
#in the form of it's item name with count
#Let us sepearte out items and their counts
items=list(reversed([i[0] for i in item_fre]))
#Thid is list comprehenasion for rach item in item fre 
#access the key
#Here you will get items list
fre=list(reversed([i[1] for i in item_fre])) 
#Here you eill get count of purchase of each item


#Now let us plot bar graph of item fre
import matplotlib.pyplot as plt
#Here we are taking fre from zero to 11 ,
#You can try 0-15 or any other
plt.bar(height=fre[0:11],x=list(range(0,11)))
plt.xticks(list(range(0,11)),items[0:11])
#plt.xticks you can specify a rotation for the tick
#labels in degrees or with keywords
plt.xlabel("items")
plt.ylabel("count")
plt.show()
import pandas as pd
#Now let us try to establish association rule mining
#we have groceries list in the list format,
#we need to convert it in dataframe
gro_series=pd.DataFrame(pd.Series(gro_list))
#Now we wwill get dataframe of size 9836*1 size,col 
#comprises of multpal items 
#we had extra row creaetd ,check the gro_series,
#last row is empty let us first delete it
gro_series=gro_series.iloc[:9835,:]
#We have taken rows from 0 to 9834 and  col 0 to all
#groceries series has col having name 0 let us rename as transactions
gro_series.columns=["Transactions"]
#Now we will have to aaply one hot encoding before that in 
#one col there are various items sepearted by ,

#',' let us seperate it with '*'
x=gro_series['Transactions'].str.join(sep='*')
#check the x in variable exploreer which has * separetorrather than ,
x=x.str.get_dummies(sep='*')
#You eill get one hot encoded dataframe of size 9835*169
#This is our input data to apply to aprioiri alg ,it will
#generste !169
#is 0.0075 (it must be between 0 to 1)
#You can give any number but must be between
fre_itemsets=apriori(x,min_support=0.0075,max_len=4,use_colnames=True)
#You will get support values for 1,2,3,4  and max items
#let us sort these support values
fre_itemsets.sort_values('support',ascending=False,inplace=True)
#support values will be sortded in des order
#Even EDA was also have the same trend ,in EDA there was count
#and here it is support values
#We will generate association rule,This association rule
#will calculate all the matrix
#Of each and every combinaation
rules=association_rules(fre_itemsets,metric='lift',min_threshold=1)
#This generate association rule of size 1198*9 col
#Comprizes of antescends,consepseusne
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)
