from sklearn.neural_network import  MLPClassifier, MLPRegressor
from helper_functions import *
import numpy as np
import pandas as pd 
# Matplotlib for visualizing graphs
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
# Sklearn for creating a dataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Set parameters for plotting
params = {'axes.titlesize': 'xx-large',               # Set title size
          'axes.labelsize': 'x-large',                # Set label size
          'figure.figsize': (8, 6)                    # Set a figure Size
}
rcParams.update(params)

# Sample size
#M = 200
# No. of input features
#n = 1
# Learning Rate
#l_r = 0.05
# Number of iterations for updates
#epochs = 51

#X, y = make_regression(n_samples=M, n_features=n, n_informative=n, n_targets=1, random_state=42, noise=10)
def plot_graph(X, y):
     # Plot the original set of datapoints
    _ = plt.scatter(X, y, alpha=0.8)
    _ = plt.title('Plot of Datapoints generated')
    _ = plt.xlabel('x')
    _ = plt.ylabel('y')
    plt.show()

#plot_graph(X, y)

#print('Shape of vector X:', X.shape)
#print('Shape of vector y:', y.shape)
#print(y)

def reset_sizes(*args):
    return tuple(arg.reshape((arg.shape[0], 1)) for arg in args)

X, y = reset_sizes(X, y)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Function to generate parameters of the linear regression model, m & b.
def init_params():
    m = np.random.normal(scale=10)
    b = np.random.normal(scale=10)
    return m, b

# Call function to generate paramets
m, b = init_params()

def plot_graph(dataset, pred_line=None):
    X, y = dataset['X'], dataset['y']
    # Plot the set of datapoints
    _ = plt.scatter(X, y, alpha=0.8)                                
    if(pred_line != None):
        x_line, y_line = pred_line['x_line'], pred_line['y_line']
        # Plot the randomly generated line
        _ = plt.plot(x_line, y_line, linewidth=2, markersize=12, color='red', alpha=0.8)
        _ = plt.title('Random Line on set of Datapoints')
    else:
        _ = plt.title('Plot of Datapoints')
    _ = plt.xlabel('x')
    _ = plt.ylabel('y')
    plt.show()

# Function to plot predicted line
def plot_pred_line(X, y, m, b):
    # Generate a set of datapoints on x for creating a line.
    x_line = np.linspace(np.min(X), np.max(X), 10)
    # Calculate the corresponding y with random values of m & b
    y_line = m * x_line + b
    dataset = {'X': X, 'y': y}
    pred_line = {'x_line': x_line, 'y_line':y_line}
    plot_graph(dataset, pred_line)
    return

plot_pred_line(X_train, y_train, m, b)

def forward_prop(X, m, b):
    y_pred = m * X + b
    return y_pred

y_pred = forward_prop(X_train, m, b)

'''
filepath_data_1000 = 'C:/Users/Kathan/Desktop/AML/Assignment4/dataset/x1000.mat'
filepath_data_10k = 'c:/Users/Kathan/Desktop/AML/Assignment4/dataset/x10k.mat'
train= mat_to_array(filepath_data_1000,0,1000,'x1000')
test = mat_to_array(filepath_data_10k,0,10000,'x10k')

x_train = np.reshape(train[:,0],(-1,1))
y_train = train[:,1]
x_test = np.reshape(test[:,0],(-1,1))
y_test = test[:,1]

'''















'''
#,alpha=0.1, learning_rate="constant",max_iter=1000
reg = MLPRegressor(6,activation="tanh",max_iter=5000,solver='sgd')
reg.fit(x_train,y_train)
y_test_predict = reg.predict(x_test)

#score = reg.score(x_test, y_test)

#print(score)


x_train = np.reshape(x_train1[:,0],(-1,1))
y_train = np.reshape(x_train1[:,1],(-1,1))
x_test = np.reshape(x_test1[:,0],(-1,1))
y_test = np.reshape(x_test1[:,1],(-1,1))
clf = MLPClassifier(1,activation="identity",alpha = 0.01,learning_rate="constant",max_iter=10)
'''

#clf.fit(x_train,y_train)
#score = clf.score(x_test, y_test)

#print(x_train)
#print(y_train)


#print(np.shape(x))
#print(np.shape(y))

