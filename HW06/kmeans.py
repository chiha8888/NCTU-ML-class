import numpy as np
from util import *

EPS=1e-9

def initial_mean(X,k,initType):
    '''
    @param X: (#datapoint,#features) ndarray
    @param k: #clusters
    @param initType: 'random','pick','k_means_plusplus'
    @return: (k,#features) ndarray, Kij: cluster i's j-dim value
    '''
    Cluster = np.zeros((k, X.shape[1]))
    if initType == 'k_means_plusplus':
        # reference: https://www.letiantian.me/2014-03-15-kmeans-kmeans-plus-plus/
        #pick 1 cluster_mean
        Cluster[0]=X[np.random.randint(low=0,high=X.shape[0],size=1),:]
        #pick k-1 cluster_mean
        for c in range(1,k):
            Dist=np.zeros((len(X),c))
            for i in range(len(X)):
                for j in range(c):
                    Dist[i,j]=np.sqrt(np.sum((X[i]-Cluster[j])**2))
            Dist_min=np.min(Dist,axis=1)
            sum=np.sum(Dist_min)*np.random.rand()
            for i in range(len(X)):
                sum-=Dist_min[i]
                if sum<=0:
                    Cluster[c]=X[i]
                    break
    elif initType=='pick':
        random_pick=np.random.randint(low=0,high=X.shape[0],size=k)
        Cluster=X[random_pick,:]
    else: # initType=='random'
        X_mean=np.mean(X,axis=0)
        X_std=np.std(X,axis=0)
        for c in range(X.shape[1]):
            Cluster[:,c]=np.random.normal(X_mean[c],X_std[c],size=k)

    return Cluster

def kmeans(X,k,H,W,initType='random',gifPath='default.gif'):
    '''
    k clusters
    @param X: (#datapoint,#features) ndarray
    @param k: # clusters
    @param H: image H
    @param W: image W
    @return: (#datapoint) ndarray, Ci: belonging class of each data point
    @return: ndarray list ready for gif
    '''
    Mean=initial_mean(X,k,initType)

    # Classes of each Xi
    C=np.zeros(len(X),dtype=np.uint8)
    segments=[]

    diff=1e9
    count=1
    while diff>EPS :
        # E-step
        for i in range(len(X)):
            dist=[]
            for j in range(k):
                dist.append(np.sqrt(np.sum((X[i]-Mean[j])**2)))
            C[i]=np.argmin(dist)

        # M-step
        New_Mean=np.zeros(Mean.shape)
        for i in range(k):
            belong=np.argwhere(C==i).reshape(-1)
            for j in belong:
                New_Mean[i]=New_Mean[i]+X[j]
            if len(belong)>0:
                New_Mean[i]=New_Mean[i]/len(belong)

        diff = np.sum((New_Mean - Mean)**2)
        Mean=New_Mean

        # visualize
        segment = visualize(C,k,H,W)
        segments.append(segment)
        print('iteration {}'.format(count))
        for i in range(k):
            print('k={}: {}'.format(i + 1, np.count_nonzero(C == i)))
        print('diff {}'.format(diff))
        print('-------------------')
        cv2.imshow('', segment)
        cv2.waitKey(1)

        count+=1

    return C,segments
