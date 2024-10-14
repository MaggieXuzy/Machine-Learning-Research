import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import pandas as pd

df = pd.read_csv('C:/Users/xuziy/Desktop/Machine Learning/iris.csv')
data = df.iloc[:,1:5].values()

def k_means(num,mean_num): #num is the number of clusters mean_num is the number of current mean
    


