# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:19:37 2020

@author: suraj baraik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

zoo = pd.read_csv("C:\\Users\\suraj baraik\\Desktop\\Data Science\\Suraj\\New folder (12)\\Module 18 Machine Learning K nearest Neighbour\\Assignment\\Zoo.csv")


zoo.rename(columns = {"animal name":"animalname"},inplace =True)

############ EDA ###########
ani = zoo["animalname"].value_counts()
### As we look at the data,
#animal name is the column whch is categorical in nature. So, we converted into dummy variables
##if we look at all the columns, all the variables except legs are binary values in nature. 
## Legs column is actually a set of values which being repeated. So, we tend to factorize the variable.

#zoo["animalname"],_=pd.factorize(zoo["animalname"])

zoo["type"].value_counts()
data = zoo.describe()
zoo.info()

zoo["legs"].var() ### 4.13
##and even the standard deviation is 2.03
zoo["hair"].var()## 0.24
zoo["feathers"].var()## 0.16
zoo["eggs"].var() ##0.2453
zoo["milk"].var()##0.243
zoo["airborne"].var()#0.183

##Creating variance dataframe
variance = zoo.var()

##As we do describe on the data,
#The minimum of all the vaiables except legs is'0' and the maximum value of the variables except legs is 1.
##Legs minimum value is zero and maximum value is 8
##The whole data is discreate and categorical in nature with Binary data.
#The output variable is type.
##Animal name has discreate, categorical, non numeric data. So, we convert them into dummy variables.

##Plotting the boxplot and histogram
plt.boxplot(zoo["legs"])
##In Legs variable there is an outlier.
plt.boxplot(zoo["type"])
##This is the output variable. As it is discreate,categorical data(set of values being repeated) analysing from boxplot is difficult.
plt.hist(zoo["legs"])
##Again histogram for this type of data, we cannot plot the histogram. As, the whole data is in Binary form, except the Legs variable which 
## is in categorcial( set of values are being repeated) so plotting histogram doesnt make sense.


##Creating dummy variables for the animal variable 
dummy = pd.get_dummies(zoo["animalname"],drop_first =True)

zoo = pd.concat([zoo,dummy],axis=1)

zoo = zoo.drop(["animalname"],axis=1)

zoo["legs"].value_counts()
zoo["legs"],_=pd.factorize(zoo["legs"])

labels = zoo.iloc[:,16]
features = zoo.drop(["type"],axis=1)

##Normalizing the equation

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

data = norm_func(features)

##Splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size =0.2, stratify=labels)

from sklearn.neighbors import KNeighborsClassifier as KN
model1 =KN(n_neighbors = 5)
model1.fit(x_train,y_train)

##Accuracy on the training data
train_acc = np.mean((model1.predict(x_train)==y_train))
###97.5%

##Accuracy on test data
test_Acc = np.mean(model1.predict(x_test)==y_test)
###85.2%

##Trying for k=7

model2 = KN(n_neighbors=7)
model2.fit(x_train,y_train)

###Accuracy ofn training data
train2_acc = np.mean(model2.predict(x_train)==y_train)
##96.25

##Accuracy on test data

test2_acc = np.mean(model2.predict(x_test)==y_test)
##85.2%

###Creating a empty list
acc=[]

##running KNN algorithm for 7 to 50 nearest neighbours and 
# storing the accuracy values 

for i in range(7,50,2):
    model2=KN(n_neighbors = i)
    model2.fit(x_train,y_train)
    train_acc = np.mean(model2.predict(x_train)==y_train)
    test_acc = np.mean(model2.predict(x_test)==y_test)
    acc.append([train_acc,test_acc])
    
import matplotlib.pyplot as plt

##training accuracy plot
plt.plot(np.arange(7,50,2),[i[0] for i in acc],"bo-")

##test accuracy plot
plt.plot(np.arange(7,50,2),[i[1] for i in acc],"ro-")    

plt.legend(["train_acc", "test_acc"])

## The plot shows k=17,Trying for k=17

model_fin = KN(n_neighbors = 17) 
model_fin.fit(x_train,y_train)

##Accuracy on training data
train_fin = np.mean(model_fin.predict(x_train)==y_train)
###90%

##Accuracy on test data
test_fin =np.mean(model_fin.predict(x_test)==y_test)
##90.4%
