# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:47:44 2016

@author: caoloi


(From Loi Van Cao et al, Cybernetics 2018)
"""
import numpy as np
from sklearn import preprocessing

seed = 0

"Normalize training and testing sets"
def normalize_data(train_X, test_X, scale):
    if ((scale == "standard") | (scale == "maxabs") | (scale == "minmax")):
        if (scale == "standard"):
            scaler = preprocessing.StandardScaler()
        elif (scale == "maxabs"):
            scaler = preprocessing.MaxAbsScaler()
        elif (scale == "minmax"):
            scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X  = scaler.transform(test_X)
    else:
        print ("No scaler")
    return train_X, test_X


def read_pair_data(trainf, testf):
    """Given a pair of filenames, assuming y is last column."""
    d = np.genfromtxt(trainf, delimiter="\t", skip_header=1)
    Xtrain, ytrain = d[:, :-1], d[:, -1:]
    d = np.genfromtxt(testf, delimiter="\t", skip_header=1)
    Xtest, ytest = d[:, :-1], d[:, -1:]
    return Xtrain, ytrain, Xtest, ytest
    


"*************************Read data from CSV file*****************************"
def process_PenDigits():
    #16 real-values + 1 class (normal: class 0, anomaly: class 2 or 1-9)
    #each class is equaly to each others in train and test set
    d0 = np.genfromtxt("../data/pendigits_train.csv", delimiter=",")
    d1 = np.genfromtxt("../data/pendigits_test.csv", delimiter=",")

    # shuffle
    np.random.seed(seed)
    np.random.shuffle(d0)
    np.random.shuffle(d1)

    "Pre-processing training set"
    dy = d0[:,-1]                       # put labels(the last column) to dy
    train_X = d0[(dy==0)]               # Normal(class 0) to train_X
    train_X = train_X[:,0:-1]           # discard the last column (labels)

    "Pre-processing Testing set"
    dy = d1[:,-1]                       # put labels to dy
    dX = d1[:,0:-1]                     # put data to dX without last column (labels)

    dX0 = dX[(dy == 0)]                  # Normal, class 0
    dX1 = dX[(dy == 2)]                   # Anomaly, class 2 (or 1-9)

    test_X0 = dX0                       #normal test
    test_X1 = dX1                       #anomaly test
    #normal and anomaly test
    test_X = np.concatenate((test_X0, test_X1))

    #Create label for normal and anomaly test examples, and then combine two sets
    test_y0 = np.full((len(test_X0)), False, dtype=bool)
    test_y1 = np.full((len(test_X1)), True,  dtype=bool)
    test_y =  np.concatenate((test_y0, test_y1))
    #create binary label (1-normal, 0-anomaly) for compute AUC later
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual



"*****************************Load dataset*****************************"
def load_data(dataset):

    if (dataset == "ACA"):
        d = np.genfromtxt("../data/australian.csv", delimiter=",")
        label_threshold = 0
        # 14 feature + 1 class [ 0 (383 normal), 1 (307 anomaly)]
        # 8 CATEGORICAL FEATURES NEED TO BE PREPROCESSED

    elif (dataset == "GLASS"):
        d = np.genfromtxt("../data/glass.csv", delimiter=",")
        label_threshold = 4
        # 9 attributes + 1 class [1-4 - 163 window glass (normal); 5-7 - 51 Non-window glass (anomaly)]

    elif (dataset == "PenDigits"):
        train_X, test_X, actual = process_PenDigits()
        #16 real-value + 1 class attribute (0 as Normal - 2 ( or 1,2,3,4,5,6,7,8,9)
        #Training 780 normal, testing 363 normal and 364 anomaly
        return train_X, test_X, actual


    else:
        print ("Incorrect data")


        

    "*************************Chosing dataset*********************************"
    d = d[~np.isnan(d).any(axis=1)]    #discard the '?' values

    np.random.seed(seed)
    np.random.shuffle(d)

    dX = d[:,0:-1]              #put data to dX without the last column (labels)
    dy = d[:,-1]                #put label to dy

    dy = dy > label_threshold
                                # dy=True with anomaly labels
                                # separate into normal and anomaly
    dX0 = dX[~dy]               # Normal data
    dX1 = dX[dy]                # Anomaly data
    dy0 = dy[~dy]               # Normal label
    dy1 = dy[dy]                # Anomaly label

    #print("Normal: %d Anomaly %d" %(len(dX0), len(dX1)))
    split = 0.8             #split 80% for training, 20% for testing

    idx0  = int(split * len(dX0))
    idx1  = int(split * len(dX1))

    train_X = dX0[:idx0]        # train_X is 80% of the normal class

    # test set is the other half of the normal class and all of the anomaly class
    test_X = np.concatenate((dX0[idx0:], dX1[idx1:]))  # 30% of normal and 30% of anomaly
    test_y = np.concatenate((dy0[idx0:], dy1[idx1:]))  # 30% of normal and 30% of anomaly label
    #conver test_y into 1 or 0 for computing AUC later
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual

