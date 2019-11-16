import numpy as np
from sklearn.cluster import KMeans

a=np.random.random((3,5))
print('\t','HHH')


'''
M=np.random.rand(10000)
print(M)
km=KMeans(n_clusters=2).fit(M.reshape(-1,1))
print(km.labels_)
print(km.cluster_centers_)
print(np.sum(km.cluster_centers_)/2)
'''

'''
print(M)
maxs=np.argmax(M,axis=1)
print(maxs)
#M=np.asarray(M>0.5,dtype='uint8')
print(np.sum(M,axis=1))

weight=np.array([[1,2,3]])
print(weight)
print(M+weight)
#weight=np.array([[1],[2],[3]])
print(weight.reshape(-1,1))
print(M+weight.reshape(-1,1))
#print(np.multiply(M,weight))
'''