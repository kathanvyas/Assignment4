import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from helper_functions import *
import pandas as pd 
import matplotlib.pyplot as plt
#data = np.genfromtxt(r"""file location""", delimiter=',')
filepath_data_1000 = 'C:/Users/Kathan/Desktop/AML/Assignment4/dataset/x1000.mat'
filepath_data_10k = 'c:/Users/Kathan/Desktop/AML/Assignment4/dataset/x10k.mat'
train= mat_to_array(filepath_data_1000,0,1000,'x1000')
test = mat_to_array(filepath_data_10k,0,10000,'x10k')

x_train = np.reshape(train[:,0],(-1,1))
y_train = train[:,1]
x_test = np.reshape(test[:,0],(-1,1))
y_test = test[:,1]



model = Sequential()
model.add(Dense(1,input_dim = 1))
model.add(Dense(units = 6, activation = 'softmax'))
#model.add(Dense(units = 32, activation = 'softmax'))
model.add(Dense(units = 1))
#model.add(Dense(1,))
#model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])

model.compile(optimizer = 'sgd',loss = 'mean_squared_error')

model.fit(x_train, y_train, batch_size = 10, epochs = 100)


y_pred = model.predict(x_train)


plt.scatter(x_train,y_train, color = 'red', label = 'Real data')
plt.scatter(x_train,y_pred, color = 'blue', label = 'model')
plt.title('Prediction')
plt.legend()
plt.show()
#Y = data[:,-1]
#X = data[:, :-1]
