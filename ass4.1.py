from helper_functions import *
import numpy as np
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D
import os
if not os.path.isdir('ass4.1_images'):
    os.mkdir('ass4.1_images')
np.random.seed(404)

# Functions
def graph_real_and_predicted(dataset, yhat, fname=None):
    X = dataset.drop('target', 1)
    Y = dataset['target']
    X1 = X['feature']
    fig = plt.figure(dpi=80, figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.scatter(X1, Y, c='r', label='Orignal Values', s = 1)
    ax.scatter(X1, yhat, c='b', label='Proposed Model', s = 1)
    ax.set_xlabel('Feature Variable')
    ax.set_ylabel('Target Variable')
    plt.legend()
    if fname is not None:
        plt.savefig('ass4.1_images/' + fname + '.jpg')
def model_target_MLP(dataset, hidden, print_coefs = True, max_iter= 10000,x_test = 0):
    num_samples = dataset.shape[0]
    cutoff = (num_samples * 3) // 4
    Xtrn = dataset.drop('target', 1).iloc[:cutoff,:]
    Ytrn = dataset['target'].iloc[:cutoff]
    Xval = dataset.drop('target', 1).iloc[cutoff:,:]
    Yval = dataset['target'].iloc[cutoff:]
    model = MLPRegressor(hidden, validation_fraction = 0, solver='sgd', max_iter= max_iter).fit(Xtrn, Ytrn)
    coefs = model.coefs_
    yhat = model.predict(X)
    yhatval = model.predict(Xval)
    loss = np.square(Yval - yhatval).mean()
    hiddens = coefs[0].T
    final_mlp = coefs[1].flatten()
    
    Xtst = dataset_test.drop('target', 1)
    y_test_predict = model.predict(Xtst)
    
    coefs = list(zip([dict(zip(X.columns, h)) for h in hiddens],
                     [['output mult:', m] for m in  final_mlp.flatten()], 
                     [['intercept:', i] for i in  model.intercepts_[0]]))
    print('loss:', loss)
    if print_coefs:
        for idx, c in enumerate(coefs):
            f1, o, i = c
            if (idx == 1):
                print('Target', '=', f1['feature'].round(2), '* feature variable')
        output = 'yhat = '
        for fidx, v in enumerate(final_mlp):
            output = output + str(v.round(2)) + ' * feat ' + str(fidx) + ' + '
        output = output + str(model.intercepts_[1][0].round(2))
        print(output)
    return model, yhat, coefs, loss,y_test_predict
def plot_dataset(xtr,fname=None):
    feature_var = xtr[:,0]
    target = xtr[:,1]
    dataset = pd.DataFrame({'feature': feature_var, 'target': target}).round(2)
    X = dataset.drop('target', 1)
    Y = dataset['target']
    X1 = X['feature']
    fig = plt.figure(dpi=80, figsize = (10, 4))
    ax = fig.add_subplot(111)
    ax.scatter(X1, Y, c='r', label='target variable', s = 1)
    ax.set_xlabel('Feature Variable (train)')
    ax.set_ylabel('Target Variable (train)')
    plt.legend()
    plt.savefig('ass4.1_images/' + fname + '.jpg')

#Train and Test Datasets
filepath_data_1000 = 'c:/Users/Kathan/Desktop/AML/Assignment4/dataset/x1000.mat'
filepath_data_10k = 'c:/Users/Kathan/Desktop/AML/Assignment4/dataset/x10k.mat'
xtr = mat_to_array(filepath_data_1000,0,1000,'x1000')
feature_var = xtr[:,0]
target = xtr[:,1]
dataset = pd.DataFrame({'feature': feature_var, 'target': target}).round(2)
X = dataset.drop('target', 1)
Y = dataset['target']
plot_dataset(xtr,'train_data')

xte = mat_to_array(filepath_data_10k,0,10000,'x10k')
feature_var_test = xte[:,0]
target_test = xte[:,1]
dataset_test = pd.DataFrame({'feature': feature_var_test,'target': target_test}).round(2)
X_test = dataset.drop('target', 1)
Y_test = dataset['target']
plot_dataset(xte,'test_data')



model, yhat, coefs, loss, y_test_pred = model_target_MLP(dataset, hidden= [6], x_test= dataset_test)
graph_real_and_predicted(dataset, yhat, 'MLP (neural_network)')
graph_real_and_predicted(dataset_test,y_test_pred,'MLP_testdata_result')
