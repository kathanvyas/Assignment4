import h5py
import numpy as np
from random import randrange
from numpy import exp, array, random, dot
from helper_functions import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.svm import SVC

    
    
def mat_to_array(filepath,flag,N,name):
    '''
    Function that converts mat file to numpy array.
    filepath: path to the mat file.
    flag: 0-data |  1-label
    n: number of samples
    name: name of mat file ins string format
    Author: Kathan Vyas
    '''
    if flag == 0:
        x_y_numpy_array = np.zeros((N, 2), dtype=float)
    else:
        x_y_numpy_array = np.zeros((N, 1), dtype=float)

    
    with h5py.File(filepath, 'r') as f:
        for idx, element in enumerate(f[name]):
            x_y_numpy_array[idx] = element[:]
    return x_y_numpy_array
def min_P_error_classifier(sample_size,class_prior0,class_prior1,dataset,orig_label,gmean,gcov):
    
    #As it is min P(error) classifer, we will always take 0/1 loss
    loss = np.array([[0,1], [1,0]])
    size = sample_size
    prior = [class_prior0,class_prior1]
    
    mean = np.zeros((2,4)) 
    mean[:,0] = gmean[:,0] 
    mean[:,1] = gmean[:,1]
    
    cov = np.zeros((2,2,4))
    cov[:,:,0] = gcov[:,:,0]
    cov[:,:,1] = gcov[:,:,1]
    
    # Gamma/ threshold
    gamma = ((loss[1,0]-loss[0,0])/(loss[1,0] - loss[1,1])) * (prior[0]/prior[1])
    orig_labels = orig_label

    
    new_labels = np.zeros((1,size))
    # Calculation for discriminant score and decisions
    cond_pdf_class0_log = np.log((multivariate_normal.pdf(dataset.T,mean=mean[:,0],cov = cov[:,:,0])))
    cond_pdf_class1_log = np.log((multivariate_normal.pdf(dataset.T,mean=mean[:,1],cov = cov[:,:,1])))
    
    discriminant_score = cond_pdf_class1_log - cond_pdf_class0_log


    new_labels[0,:] = (discriminant_score >= np.log(gamma)).astype(int)

    # Code to plot the distribution after Classification
    x00 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 0 and new_labels[0,i] == 0)]
    x01 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 0 and new_labels[0,i] == 1)]
    x10 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 1 and new_labels[0,i] == 0)]
    x11 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 1 and new_labels[0,i] == 1)]
    plt.plot(dataset[0,x00],dataset[1,x00],'.',color ='g')
    plt.plot(dataset[0,x01],dataset[1,x01],'.',color = 'r')
    plt.plot(dataset[0,x11],dataset[1,x11],'+',color ='g')
    plt.plot(dataset[0,x10],dataset[1,x10],'+',color = 'r')
    plt.legend(["class 0 correctly classified",'class 0 wrongly classified','class 1 correctly classified','class 1 wrongly classified'])
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title('Distribution after classification')
    plt.show()
    
    
    c0 = np.argwhere(orig_labels[0,:]==0).shape[0]
    c1 = np.argwhere(orig_labels[0,:]==1).shape[0]
    #print("Class 0:",c0)
    #print("Class 1:",c1)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    tpr = 0
    fpr = 0
    min_TPR = 0
    min_FPR = 0
    TPR = []
    FPR = []
    new_labels1 = np.zeros((1,size))
    d_labels1 = np.zeros((1,size))
    r=map(lambda x: x/10.0,range(0,500))
    print(r)
    for i in r:
        gamma1 = i
        #print(gamma)
        new_labels1[0,:] = (discriminant_score >= np.log(gamma1)).astype(int)
        #d_labels1[0,:] = discriminant_score >= np.log(gamma)
        for i in range(new_labels1.shape[1]): 
            #print("innerforloop")
            if (orig_labels[0,i] == 1 and new_labels1[0,i] == 1):
               TP += 1
            if (orig_labels[0,i] == 0 and new_labels1[0,i] == 1):
               FP += 1
            if (orig_labels[0,i] == 0 and new_labels1[0,i] == 0):
               TN += 1
            if (orig_labels[0,i] == 1 and new_labels1[0,i] == 0):
               FN += 1
        tpr = TP / (TP+FN)
        fpr = FP / (FP+TN)
        TPR.append(tpr)
        FPR.append(fpr)
        if gamma1 == 9.00000:
            min_TPR = tpr
            min_FPR = fpr
        

    plt.plot(FPR,TPR,'-',color = 'r')
    plt.plot(min_FPR,min_TPR, 'g*')
    plt.legend(["ROC Curve",'Min P Error'])
    plt.show()
    plt.close()
def cross_validation_split(dataset, folds=3):
    '''
    Function performs k-fold cross validation
    dataset: numpy array
    folds: number of folds
    Author: Kathan Vyas
    '''
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def check_splits_svm(data):
    C = []
    S = []
    SC = []
    c = 1
    sigma = 0.5
    for d in data[0]:
       clf =  SVC(gamma=sigma, C=c)
       clf.fit(data[:,d,0],data[:,:,-1])
       score = clf.score(data[:,d,0],data[:,:,-1])
       C.append(c)
       S.append(sigma)
       SC.sppend(score)
       
    
       