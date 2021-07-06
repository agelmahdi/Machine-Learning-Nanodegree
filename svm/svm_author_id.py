#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import svm

# use a linear kernel
# clf = svm.LinearSVC()

# use a RBF kernel
# Optimize C Parameter
clf = svm.SVC(kernel="rbf", C=10000.)

# slice the training dataset down to 1% of its original size
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train,labels_train)

pred = clf.predict(features_test)

# Extracting Predictions from an SVM

print pred[10]
print pred[26]
print pred[50]

count =0
for p in pred :
    if p == 1 :
        count += 1

print count




# from sklearn.metrics import accuracy_score
#
# print accuracy_score(pred,labels_test)


