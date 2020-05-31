#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:34:27 2020

@author: tarasalman
"""


import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

all = 100
attacks= int(0.2*all)
benign= int(0.8*all)
y_test=np.array([0,1,0,0,0,1,0,0,0,0]*10)


##Codes for F1Score
##Change the plot for risk, balanced accuracy and accuracy

x=np.array(range(attacks,0,-1))
TP=x
FN=attacks-x
FP=0
TN=benign-FP
x=np.array(range(0,100,5))
fig= plt.figure(figsize=(2,2))

##NO error
risk=(0.2*TP+0.1*TN)/(FP+10*FN+0.2*TP+0.1*TN)
accuracy=(TP+TN)/(FP+FN+TP+TN)
bal_acc=((TP)/(TP+FN)+(TN)/(TN+FP))/2
precision=(0.2*TP)/(0.2*TP+FP)
recall=(0.2* TP)/(0.2*TP+10*FN)
precision=(TP)/(TP+FP)
recall=( TP)/(TP+FN)
f1score=2*(precision*recall)/(precision+recall)

plt.plot(x,f1score,'b',label='0% FP')

# 5% error
FP=benign*0.05
TN=benign-FP

bal_acc=((TP)/(TP+FN)+(TN)/(TN+FP))/2
accuracy=(TP+TN)/(FP+FN+TP+TN)
risk=(0.1*TP+0.2*TN)/(FP+10*FN+0.1*TP+0.2*TN)
precision=(0.2*TP)/(0.2*TP+FP)
recall=(0.2* TP)/(0.2*TP+10*FN)
precision=(TP)/(TP+FP)
recall=( TP)/(TP+FN)
f1score=2*(precision*recall)/(precision+recall)
plt.plot(x,f1score,'k',label='5% FP')

#10 %Error
FP=benign*0.10
TN=benign-FP

bal_acc=((TP)/(TP+FN)+(TN)/(TN+FP))/2
accuracy=(TP+TN)/(FP+FN+TP+TN)
risk=(0.1*TP+0.2*TN)/(FP+10*FN+0.1*TP+0.2*TN)
precision=(0.2*TP)/(0.2*TP+FP)
recall=(0.2* TP)/(0.2*TP+10*FN)
precision=(TP)/(TP+FP)
recall=( TP)/(TP+FN)
f1score=2*(precision*recall)/(precision+recall)
plt.plot(x,f1score,'C1',label='10% FP')

#Plot others
plt.xticks(np.array(range(0,100,20)))
plt.xlabel('Percentage of FN (FN/(FN+TN))')
plt.ylim([0,1])
plt.ylabel('F1score')

plt.legend()


##Codes for AUC

fig= plt.figure(figsize=(2,2))
y_test=np.array([0,1,0,0,0,1,0,0,0,0]*10)
y_predicted=np.array([0,1,0,0,0,1,0,0,0,0]*10)
auc_f=[]
for i in range (0,20):
    if i<10:
        y_predicted[1+i*10]=0
    else:
        y_predicted[5+(i-10)*10]=0
    auc=metrics.roc_auc_score(y_test, y_predicted)
    auc_f.append(auc)
  #  print(auc)
plt.plot(x,auc_f,'b',label='0% FP')

y_test=np.array([0,1,0,0,0,1,0,0,0,0]*10)
y_predicted=np.array([0,1,0,0,0,1,0,0,0,0]*10)
auc_f=[]

for i in range (0,20):
    if i<10:
        y_predicted[1+i*10]=0
    else:
        y_predicted[5+(i-10)*10]=0
    if i<4: 
        y_predicted[0+i*10]=1
    auc=metrics.roc_auc_score(y_test, y_predicted)
    auc_f.append(auc)
 #   print(auc)
plt.plot(x,auc_f,'k',label='5% FP')
    
y_test=np.array([0,1,0,0,0,1,0,0,0,0]*10)
y_predicted=np.array([0,1,0,0,0,1,0,0,0,0]*10)
auc_f=[]

for i in range (0,20):
    if i<10:
        y_predicted[1+i*10]=0
    else:
        y_predicted[5+(i-10)*10]=0
    if i<8: 
        y_predicted[0+i*10]=1
    auc=metrics.roc_auc_score(y_test, y_predicted)
    auc_f.append(auc)
 #   print(auc)
plt.plot(x,auc_f,'C1',label='10% FP')
plt.xticks(np.array(range(0,100,20)))
plt.xlabel('Percentage of FN (FN/(FN+TN))')
plt.ylim([0,1])
plt.ylabel('AUC')
plt.legend()
    






