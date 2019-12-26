import numpy as np
import cv2
from scipy.spatial.distance import pdist,squareform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from array2gif import write_gif

# predefine colormap
colormap= np.random.choice(range(256),size=(100,3))
#colormap=np.asarray([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[255,255,255]],dtype=np.uint8)

def imread(path):
    '''
    @param path:
    @return: (H*W,C) flatten_image ndarray
    '''
    image = cv2.imread(path)
    H, W, C = image.shape
    image_flat = np.zeros((W * H, C))
    for h in range(H):
        image_flat[h * W:(h + 1) * W] = image[h]

    return image_flat,H,W

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
    colors= colormap[:k,:]
    res=np.zeros((H,W,3))
    for h in range(H):
        for w in range(W):
            res[h,w,:]=colors[X[h*W+w]]

    return res.astype(np.uint8)

def plot_eigenvector(xs,ys,zs,C):
    '''
    only for 3-dim datas
    @param xs: (#datapoint) ndarray
    @param ys: (#datapoint) ndarray
    @param zs: (#datapoint) ndarray
    @param C: (#datapoint) ndarray, belonging class
    '''
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    markers=['o','^','s']
    for marker,i in zip(markers,np.arange(3)):
        ax.scatter(xs[C==i],ys[C==i],zs[C==i],marker=marker)

    ax.set_xlabel('eigenvector 1st dim')
    ax.set_ylabel('eigenvector 2nd dim')
    ax.set_zlabel('eigenvector 3rd dim')
    plt.show()

def save_gif(segments,gif_path):
    for i in range(len(segments)):
        segments[i] = segments[i].transpose(1, 0, 2)
    write_gif(segments, gif_path, fps=2)