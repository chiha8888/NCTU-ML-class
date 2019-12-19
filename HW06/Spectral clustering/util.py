import numpy as np
import cv2
from scipy.spatial.distance import pdist,squareform

def imread(path):
    '''
    @param path:
    @return: (H*W,C) flatten_image ndarray
    '''
    image = cv2.imread('image1.png')
    HEIGHT, WIDTH, C = image.shape
    image_flat = np.zeros((WIDTH * HEIGHT, C))
    for h in range(HEIGHT):
        image_flat[h * WIDTH:(h + 1) * WIDTH] = image[h]

    return image_flat

def precomputed_kernel(X,gamma_s=1,gamma_c=1):
    '''
    kernel function: k(x,x')= exp(-r_s*||S(x)-S(x')||**2)* exp(-r_c*||C(x)-C(x')||**2)
    :@param X: (H*W=10000,rgb=3) ndarray
    :@param gamma_s: gamma of spacial
    :@param gamma_c: gamma of color
    :@return : (10000,10000) ndarray
    '''
    n=len(X)
    # S(x) spacial ingormation
    S=np.zeros((n,2))
    for i in range(n):
        S[i]=[i//100,i%100]
    K=squareform(np.exp(-gamma_s*pdist(S,'sqeuclidean')))*squareform(np.exp(-gamma_c*pdist(X,'sqeuclidean')))

    return K

def visualize(X,k,H,W):
    '''
    @param X: (10000) belonging classes ndarray
    @param k: #clusters
    @param H: image_H
    @param W: image_W
    @return : (H,W,3) ndarray
    '''
    colors= np.random.choice(range(256),size=(k,3))
    res=np.zeros((H,W,3))
    for h in range(H):
        for w in range(W):
            res[h,w,:]=colors[X[h*W+w]]

    return res.astype(np.uint8)