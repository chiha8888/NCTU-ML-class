import numpy as np

EPS=1e-9

def initial(k):
    '''
    @param k:
    @return: (k,k) ndarray, Kij: cluster i's j-dim value
    '''
    return np.random.rand(k,k)

def kmeans(X,k):
    '''
    k clusters
    @param X: (10000,k) ndarray
    @param k:
    @return:
    '''
    # normalize
    X= (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))

    Mean=initial(k)
    # Classes of each Xi
    C=np.zeros(len(X))
    diff=1e9

    while diff>EPS :
        # E-step
        for i in range(len(X)):
            dist=[]
            for j in range(k):
                dist.append(np.sum((X[i]-Mean[j])**2))
            C[i]=np.argmin(dist)

        # M-step
        New_Mean=np.zeros(Mean.shape)
        for i in range(k):
            belong=np.argwhere(C==i).reshape(-1)
            for j in belong:
                New_Mean[i]=New_Mean[i]+X[j]
            if len(belong)>0:
                New_Mean[i]=New_Mean[i]/len(belong)

        new_diff=np.sum((New_Mean-Mean))
        if abs(new_diff-diff)<EPS:
            break
        diff= new_diff

    return C.astype(int)
