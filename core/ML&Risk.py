import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler 
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout

import scipy.stats
from sklearn.cluster import KMeans
import pickle
from random import *
from scipy.spatial import distance
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


#Change Dataset when needed
df_all_t =pd.read_csv('../datasets/DoS_all.csv')


#For pre processing if needed
# le = preprocessing.LabelEncoder()
# le.fit(df_all_t['state'])
# df_all_t['state']=le.transform(df_all_t['state'])
# df_all_t=df_all_t.drop(['state'],axis=1)
#df_all_t=df_all_t.drop(['ltime'],axis=1)


X2=df_all_t.iloc[:,0:39]
y2=df_all_t.iloc[:,39]
y2=y2.replace(-1,0)
weighted_acc_t_RF=[]
bal_acc_t_RF=[]
weighted_bal_acc_t_RF=[]
acc_t_RF=[]
f1_score_t_RF=[]
weighted_f1score_RF=[]

weighted_acc_t_KNN=[]
bal_acc_t_KNN=[]
weighted_bal_acc_t_KNN=[]
acc_t_KNN=[]
f1_score_t_KNN=[]
weighted_f1score_KNN=[]

weighted_acc_t_SVM=[]
bal_acc_t_SVM=[]
weighted_bal_acc_t_SVM=[]
acc_t_SVM=[]
f1_score_t_SVM=[]
weighted_f1score_SVM=[]

weighted_acc_t_DT=[]
bal_acc_t_DT=[]
weighted_bal_acc_t_DT=[]
acc_t_DT=[]
f1_score_t_DT=[]
weighted_f1score_DT=[]

runs=1
for i in range (0,runs):
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2)
    #Scaling 
    ss=StandardScaler()
    ss.fit(X_train)
    X_train=ss.transform(X_train)
    X_test=ss.transform(X_test)
    #Smote oversampling
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    # Random Forest
    model_RF_t = RandomForestClassifier()
    model_RF_t.fit(X_train, y_train)
    y_probs_RF_t= model_RF_t.predict_proba(X_test)
    y_predicted_RF_t = model_RF_t.predict(X_test)
    # Performance of RF
    acc_t_RF.append(metrics.accuracy_score(y_test, y_predicted_RF_t))
    bal_acc_t_RF.append(metrics.balanced_accuracy_score(y_test, y_predicted_RF_t))
    f1_score_t_RF.append(metrics.f1_score(y_test, y_predicted_RF_t))
    weights=np.where(np.logical_and(y_test==1,y_predicted_RF_t==0),10, np.where(np.logical_and(y_test==0,y_predicted_RF_t==1),1, np.where(np.logical_and(y_test==1,y_predicted_RF_t==1),0.2,0.1)))
    weights=weights/sum(weights)
    weighted_acc_t_RF.append(metrics.accuracy_score(y_test, y_predicted_RF_t,True,weights))
    weighted_bal_acc_t_RF.append(metrics.balanced_accuracy_score(y_test, y_predicted_RF_t,weights))
    weighted_f1score_RF.append(metrics.f1_score(y_test, y_predicted_RF_t,sample_weight=weights))
    # #KNN
    model_KNN = KNeighborsClassifier(n_jobs=-1).fit(X_train, y_train) #class_weight='balanced'
    y_predicted_KNN= model_KNN.predict(X_test)
    y_prob_KNN = model_KNN.predict_proba(X_test)
    #Performance of KNN
    acc_t_KNN.append(metrics.accuracy_score(y_test, y_predicted_KNN))
    bal_acc_t_KNN.append(metrics.balanced_accuracy_score(y_test, y_predicted_KNN))
    f1_score_t_KNN.append(metrics.f1_score(y_test, y_predicted_KNN))
    weights=np.where(np.logical_and(y_test==1,y_predicted_KNN==0),10, np.where(np.logical_and(y_test==0,y_predicted_KNN==1),1, np.where(np.logical_and(y_test==1,y_predicted_KNN==1),0.2,0.1)))
    weights=weights/sum(weights)
    weighted_acc_t_KNN.append(metrics.accuracy_score(y_test, y_predicted_KNN,True,weights))
    weighted_bal_acc_t_KNN.append(metrics.balanced_accuracy_score(y_test, y_predicted_KNN,weights))
    weighted_f1score_KNN.append(metrics.f1_score(y_test, y_predicted_KNN,sample_weight=weights))
    # #SVM
    # model_SVM = SVC(gamma='scale', random_state=0, probability=True).fit(X_train, y_train) #class_weight='balanced'
    # y_predicted_SVM = model_SVM.predict(X_test)
    # y_prob_SVM = model_SVM.predict_proba(X_test)
    # #Performance of SVM
    # acc_t_SVM.append(metrics.accuracy_score(y_test, y_predicted_SVM))
    # bal_acc_t_SVM.append(metrics.balanced_accuracy_score(y_test, y_predicted_SVM))
    # f1_score_t_SVM.append(metrics.f1_score(y_test, y_predicted_SVM))
    # weights=np.where(np.logical_and(y_test==1,y_predicted_SVM==0),10, np.where(np.logical_and(y_test==0,y_predicted_SVM==1),1, np.where(np.logical_and(y_test==1,y_predicted_SVM==1),0.2,0.1)))
    # weights=weights/sum(weights)
    # weighted_acc_t_SVM.append(metrics.accuracy_score(y_test, y_predicted_SVM,True,weights))
    # weighted_bal_acc_t_SVM.append(metrics.balanced_accuracy_score(y_test, y_predicted_SVM,weights))
    # weighted_f1score_SVM.append(metrics.f1_score(y_test, y_predicted_SVM,sample_weight=weights))
    
    # #NN 
    # model_NN = MLPClassifier()
    # model_NN.fit(X_train, y_train)
    # y_probs_NN=model_NN.predict_proba(X_test)
    # y_predicted_NN = model_NN.predict(X_test)
    # #Performance of ANN
    # acc_t_NN.append(metrics.accuracy_score(y_test, y_predicted_NN))
    # bal_acc_t_NN.append(metrics.balanced_accuracy_score(y_test, y_predicted_NN))
    # f1_score_t_NN.append(metrics.f1_score(y_test, y_predicted_NN))
    # weights=np.where(np.logical_and(y_test==1,y_predicted_NN==0),10, np.where(np.logical_and(y_test==0,y_predicted_NN==1),1, np.where(np.logical_and(y_test==1,y_predicted_NN==1),0.2,0.1)))
    # weights=weights/sum(weights)
    # weighted_acc_t_NN.append(metrics.accuracy_score(y_test, y_predicted_NN,True,weights))
    # weighted_bal_acc_t_NN.append(metrics.balanced_accuracy_score(y_test, y_predicted_NN,weights))
    # weighted_f1score_NN.append(metrics.f1_score(y_test, y_predicted_NN,sample_weight=weights))

    # #Decision Tree
    model_DT = DecisionTreeClassifier(max_depth=15)
    model_DT.fit(X_train, y_train)
    y_probs_DT= model_DT.predict_proba(X_test)
    y_predicted_DT= model_DT.predict(X_test)
    #Performance of Decision Tree
    acc_t_DT.append(metrics.accuracy_score(y_test, y_predicted_DT))
    bal_acc_t_DT.append(metrics.balanced_accuracy_score(y_test, y_predicted_DT))
    f1_score_t_DT.append(metrics.f1_score(y_test, y_predicted_DT))
    weights=np.where(np.logical_and(y_test==1,y_predicted_DT==0),10, np.where(np.logical_and(y_test==0,y_predicted_DT==1),1, np.where(np.logical_and(y_test==1,y_predicted_DT==1),0.2,0.1)))
    weights=weights/sum(weights)
    weighted_acc_t_DT.append(metrics.accuracy_score(y_test, y_predicted_DT,True,weights))
    weighted_bal_acc_t_DT.append(metrics.balanced_accuracy_score(y_test, y_predicted_DT,weights))
    weighted_f1score_DT.append(metrics.f1_score(y_test, y_predicted_DT,sample_weight=weights))

x_axis=list(range(0,runs))
print(np.mean(acc_t_RF)*100)
print(np.mean(weighted_acc_t_RF)*100)
print(np.mean(f1_score_t_RF)*100)
print(np.mean(weighted_f1score_RF)*100)
print(np.mean(bal_acc_t_RF)*100)
print(np.mean(weighted_bal_acc_t_RF)*100)

print(np.mean(acc_t_KNN)*100)
print(np.mean(weighted_acc_t_KNN)*100)
print(np.mean(f1_score_t_KNN)*100)
print(np.mean(weighted_f1score_KNN)*100)
print(np.mean(bal_acc_t_KNN)*100)
print(np.mean(weighted_bal_acc_t_KNN)*100)

# print(np.mean(acc_t_SVM)*100)
# print(np.mean(weighted_acc_t_SVM)*100)
# print(np.mean(f1_score_t_SVM)*100)
# print(np.mean(weighted_f1score_SVM)*100)
# print(np.mean(bal_acc_t_SVM)*100)
# print(np.mean(weighted_bal_acc_t_SVM)*100)

print(np.mean(acc_t_DT)*100)
print(np.mean(weighted_acc_t_DT)*100)
print(np.mean(f1_score_t_DT)*100)
print(np.mean(weighted_f1score_DT)*100)
print(np.mean(bal_acc_t_DT)*100)
print(np.mean(weighted_bal_acc_t_DT)*100)

