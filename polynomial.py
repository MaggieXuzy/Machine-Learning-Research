import numpy as np
import matplotlib.pyplot as plt
from numpy import random

data = np.load('C:/Users/xuziy/Desktop/Machine Learning/regression_data.npy')
train = data['train']
test = data['test']

x_axis = np.arange(1,15)
y_axis = []
z_axis = []

#n is the degree of polynomial
#m is the number of data

data_in = train
for n in range(1,15):
    x = data_in[:,0]  #ndarray (m,)
    y = data_in[:,1]  #ndarray (m,)

    x = x.repeat(n) #[x1,x1,x1....xm,xm,xm]
    x = x.reshape(int(len(x) / n),n)
    '''x:
    x1,x1,x1...
    x2,x2,x2...
    ...
    xm,xm,xm...
    '''
    power = np.arange(n)
    x = x ** power
    w = (np.linalg.inv((x.T)@x))@(x.T)@y
    print("W")
    print(w)

    w = w[::-1]
    yw = np.poly1d(w)
    t = np.linspace(0,6.5,200)
    print(yw)

    '''
    #plot the graph both test data and train data
    plt.plot(t,yw(t),"g" + "-")
    plt.plot(train[:,0], train[:,1],"r" + 'x', label = "training data")
    plt.legend()
    plt.show(block=False)
    plt.pause(1)
    plt.show()'''

    x = data_in[:,0]   
    loss = (y - yw(x))@(y - yw(x))
    print(loss)
    y_axis.append(loss)

#test data loss function
data_in = test
for n in range(1,15):
    x = data_in[:,0]  #ndarray (m,)
    y = data_in[:,1]  #ndarray (m,)

    x = x.repeat(n) #[x1,x1,x1....xm,xm,xm]
    x = x.reshape(int(len(x) / n),n)

    '''
    x1,x1,x1...
    x2,x2,x2...
    ...
    xm,xm,xm...
    '''
    
    power = np.arange(n)
    x = x ** power
    w = (np.linalg.inv((x.T)@x))@(x.T)@y
    print("W")
    print(w)

    w = w[::-1]
    yw = np.poly1d(w)
    

    t = np.linspace(0,6.5,200)

    
    '''#plot the graph both test data and train data
    plt.plot(t,yw(t),"g" + "-")
    plt.plot(test[:,0], test[:,1], "b" + 'x', label = "testing data")
    plt.legend()
    plt.show(block=False)
    plt.pause(1)
    plt.show()'''

    x = data_in[:,0] 
    loss = (y - yw(x))@(y - yw(x))
    print("loss:",loss)
    z_axis.append(loss)

#graph of loss vs n
plt.plot(x_axis,y_axis,"g" + "-",label = "training loss")
plt.plot(x_axis,z_axis,"r" + "-", label = "testing loss")
plt.legend()
plt.show()