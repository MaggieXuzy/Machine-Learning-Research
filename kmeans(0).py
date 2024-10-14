import numpy as np
import matplotlib.pyplot as plt
from numpy import random

data = np.load('C:/Users/xuziy/Desktop/Machine Learning/clustering_data.npy')
data=data['data']

print(data)

sumofvariance=[]

def naive(num,xmax,ymax):
    x=data[:,0]
    y=data[:,1]
    xmax=max(x)
    ymax=max(y)
    for i in range(num):
        mean[i]=[np.random.uniform(0,xmax),np.random.uniform(0,ymax)]

def forgy(num):
    length=len(data)

    for i in range(num):
        randomnum=int(np.random.random_integers(0,length-1))
        mean[i]=data[randomnum].tolist()

def k_means(num,mean_num):
    length=len(data)
    randomnum=int(np.random.randint(0,length-1))
    mean[0]=data[randomnum].tolist()
    
    for i in range(1,num):
        #calculate distance between all points and the exist mean
        probability_list=[]
        for j in range(length):
            distance=[]
            for z in range(mean_num):
                dis=np.linalg.norm(data[j]-mean[z])
                distance.append(dis)
            probability_list.append(int((min(distance))**2))
        probability_sum=int(sum(probability_list))

        for m in range(len(probability_list)):
            probability_list[m]/=probability_sum

        number=random.random()
        accumulatenum=0
        for j in range(len(probability_list)):
            accumulatenum+=probability_list[j]
            if accumulatenum>number:
                mean[i]=data[j].tolist()
                break
                
        mean_num+=1

def var():
    global variance
    for m in range(k):
        temp=np.array(cluster[m])
        variance+=np.var(temp)*len(temp)
    sumofvariance.append(variance)

def WCSSD():
    print(sumofvariance)
    x_axis=np.arange(1,9)
    plt.title('WCSSD agianst K')
    plt.xlabel('K')
    plt.ylabel('WCSSD')
    plt.plot(x_axis,sumofvariance)
    plt.show()

for k in range(4,5):
    mean=np.zeros((k,2))

    #naive(k,max_x,max_y)

    #forgy(k)
    
    k_means(k,1)

    print('mean:',mean)
    clusterchange=True
    time=0

    while clusterchange:
        time+=1
        clusterchange=False
        cluster = [[] for t in range(k)]
        print(cluster)

        for i in range(len(data)):
            dis=[]
            for j in range(len(mean)):
                distance=np.linalg.norm(data[i]-mean[j])
                dis.append(distance)
            cluster[dis.index(min(dis))].append(data[i].tolist())

        colour=['r','b','g','y','k','c','m','w']

        for i in range(len(cluster)):
            temp=np.array(cluster[i])
            if temp==[]:
                plt.plot(mean[i][0],mean[i][1],colour[i]+'x')
            else:                
                data_x=temp[:,0]
                data_y=temp[:,1]

                plt.plot(data_x,data_y,colour[i]+'.')
                plt.plot(mean[i][0],mean[i][1],colour[i]+'x')

                new_mean_x=sum(data_x)/len(data_x)
                new_mean_y=sum(data_y)/len(data_y)

            if round(new_mean_x,2)!=round(mean[i][0],2) or round(new_mean_y,2)!=round(mean[i][1],2):
                clusterchange=True
            mean[i][0]=new_mean_x
            mean[i][1]=new_mean_y

        print('mean:',mean)
        plt.show(block=False)
        plt.pause(1)
        plt.clf()
    
    variance=0
    #calculate variance
    #var()

#WCSSD()

time-=1
print(time)
