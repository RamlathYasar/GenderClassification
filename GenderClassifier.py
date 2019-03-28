# Classification Model

X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],
     [159,55,37],[171,75,42],[181,85,43]]
Y = ['male','female','female','female','male','male','male','female','male','female','male']
X_test = [[190,70,43],[154, 75, 42],[181,65,40]]
Y_test =['male','male','male']

import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
clf_DT = DecisionTreeClassifier()
clf_DT = clf_DT.fit(X,Y)
     
DT_pred = clf_DT.predict(X_test)
print(DT_pred)
DT_acc = accuracy_score(DT_pred,Y_test)

# logistic 
from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression()
clf_LR = clf_LR.fit(X,Y)
     
LR_pred = clf_LR.predict(X_test)
print(LR_pred)
LR_acc = accuracy_score(LR_pred,Y_test)

#K_nn 
from sklearn.neighbors import KNeighborsClassifier
clf_KNN = KNeighborsClassifier()
clf_KNN = clf_KNN.fit(X,Y)

KNN_pred = clf_KNN.predict(X_test)
print(KNN_pred)
KNN_acc = accuracy_score(KNN_pred,Y_test)

#SVM
from sklearn.svm import SVC
clf_SVM =SVC(kernel ='linear')
clf_SVM = clf_SVM.fit(X,Y)

SVM_pred = clf_SVM.predict(X_test)
print(SVM_pred)
SVM_acc = accuracy_score(SVM_pred,Y_test)

# Kernel svm
from sklearn.svm import SVC
clf_KSVM =SVC(kernel ='rbf')
clf_KSVM = clf_KSVM.fit(X,Y)

KSVM_pred = clf_KSVM.predict(X_test)
print(KSVM_pred)
KSVM_acc = accuracy_score(KSVM_pred,Y_test)

#naive bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB = clf_NB.fit(X,Y)

NB_pred = clf_NB.predict(X_test)
print(NB_pred)
NB_acc = accuracy_score(NB_pred,Y_test)

# Random forest
from sklearn.ensemble import RandomForestClassifier
clf_RF =RandomForestClassifier()
clf_RF = clf_RF.fit(X,Y)

RF_pred = clf_RF.predict(X_test)
print(RF_pred)
RF_acc = accuracy_score(RF_pred,Y_test)

# COMPARISON OF ClASSIFIER
classifiers = {0: 'decision_tree' , 1: 'LogisticRegression',2:'KNeighbors',3: 'SVM',4: 'KernelSVM',5:'Naive Bayes',6: 'random forest'}
accuracy = np.array([DT_acc,LR_acc,KNN_acc,SVM_acc,KSVM_acc,NB_acc,RF_acc])
max_acc = np.argmax(accuracy)
print('Best Gender Classifier is {}'.format(classifiers[max_acc]))
