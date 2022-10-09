import gc
import sys
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
from mosek.fusion import *
import random
import scipy.io as scio
import os
import copy
import pandas as pd
from  sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)

def data_real(filename):
    """
    load other bi-classification dataset

    """
    global num, dim
    data = pd.read_csv(filename, sep=',', header=None)
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    X = X.values
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    num, dim = X.shape
    y = y.values
    y = y.reshape(y.shape[0], 1)
    y[y <= 0] = -1
    return X, y

def data_real_mnist(filename,class1,class2):
    """
    load mnist csv

    """
    global num, dim
    data = pd.read_csv(filename, sep=',', header=None)
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    y = y.values
    y = y.reshape(y.shape[0], 1)
    index=((y[:,0]==class1) | (y[:,0]==class2))
    y=y[index,:]
    X = X.values
    X=X[index,:]
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    num, dim = X.shape

    y[y ==class2] = -1
    y[y == class1] = 1
    return X, y





def load_letter_mm():
    """
    letter data attack by min-max
    :return:
    """
    X_train_read = scio.loadmat('data/attackfile/mm letter210dirty_data.mat')
    X_train = X_train_read['dirty_data'].todense()
    X_train = np.array(X_train)
    y_train_read = scio.loadmat('data/attackfile/mm letter210dirty_label.mat')
    y_train = y_train_read['dirty_label']


    X_test_read = scio.loadmat('data/letter/lettersx.mat')
    X_test = X_test_read['test_data']
    X_test = X_test.todense()
    X_test = np.array(X_test)
    y_test_read = scio.loadmat('data/letter/lettersy.mat')
    y_test = y_test_read['test_label']

    return X_train,y_train,X_test,y_test

def load_letter_alfa():
    """
       letter data attack by alfa
       :return:
       """
    X_train_read = scio.loadmat('data/attackfile/alfa letter110dirty_data.mat')
    X_train = X_train_read['dirty_data'].todense()
    X_train = np.array(X_train)
    y_train_read = scio.loadmat('data/attackfile/alfa letter110dirty_label.mat')
    y_train = y_train_read['dirty_label']


    X_test_read = scio.loadmat('data/letter/lettersx.mat')
    X_test = X_test_read['test_data']
    X_test = X_test.todense()
    X_test = np.array(X_test)
    y_test_read = scio.loadmat('data/letter/lettersy.mat')
    y_test = y_test_read['test_label']

    return X_train,y_train,X_test,y_test

def read_mnist():
    """
    produce MNIST data csv
    :return:
    """
    mnist = datasets.fetch_openml('mnist_784', version=1)
    X = pd.DataFrame.to_numpy(mnist['data'])
    y = mnist['target']
    y = np.vstack(y)
    data = np.hstack((X, y))

    data = pd.DataFrame(data)
    data.to_csv('data/mnist.csv', header=None, index=None)


def load_energy():
    filename = 'data/AppEnergy_scale.csv'
    data = pd.read_csv(filename, sep=',', header=None)
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    X = X.values
    y = y.values

    n, d = X.shape
    shuffle_idx = np.random.permutation(n)
    X = X[shuffle_idx, :]
    y = y[shuffle_idx]
    y = y.reshape(y.shape[0], 1)

    intercept = int(n/2)
    X_train = X[:intercept,:]
    y_train = y[:intercept]
    X_test = X[intercept:,:]
    y_test = y[intercept:]
    return X_train,y_train,X_test,y_test