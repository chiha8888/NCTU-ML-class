import numpy as np

def add(M):
    M+=0.5



M=np.random.rand(3,3)
print(M)
#M[np.argwhere(M>0.5)]=1
M=np.asarray(M>0.5,dtype='uint8')
print(M)
'''
weight=np.array([[1,2,3]])
print(weight)
print(M+weight)
#weight=np.array([[1],[2],[3]])
print(weight.reshape(-1,1))
print(M+weight.reshape(-1,1))
#print(np.multiply(M,weight))
'''