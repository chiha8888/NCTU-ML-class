import numpy as np
import cv2
from util import *
from kmeans import kmeans

# read image
path='image1.png'
image_flat=imread(path)

# similarity matrix
W=precomputed_kernel(image_flat,gamma_s=0.01,gamma_c=0.01)
# degree matrix
D=np.diag(np.sum(W,axis=1))
L=D-W

# k clusters
k=4
'''
eigenvalue,eigenvector=np.linalg.eig(L)
np.save('eigenvalue_0.01',eigenvalue)
np.save('eigenvector_0.01',eigenvector)
'''

eigenvalue=np.load('eigenvalue_0.001.npy')
eigenvector=np.load('eigenvector_0.001.npy')
sort_index=np.argsort(eigenvalue)
# U
U=eigenvector[:,sort_index[1:1+k]]

# k-means
belonging=kmeans(U,k)
segment=visualize(belonging, k, H=100, W=100)
for i in range(k):
    print('k={}: {}'.format(i+1,np.count_nonzero(belonging==i)))

cv2.imshow('',segment)
cv2.waitKey(0)
cv2.destroyAllWindows()