import numpy as np
import os
import matplotlib.pyplot as plt
from util import imread,show_eigenface,show_reconstruction,performance
from pca import pca
EPS=0  # discard eigenvalue smaller than 0

if __name__=='__main__':
    filepath=os.path.join('Yale_Face_Database','Training')
    H,W=231,195
    X,y=imread(filepath,H,W)

    eigenvalues,eigenvectors,X_mean=pca(X)
    # Transform matrix
    U=eigenvectors.copy()
    print('U shape: {}'.format(U.shape))

    # show top 25 eigenface
    show_eigenface(U,25,H,W)

    # reduce dim (projection)
    Z=U.T@(X-X_mean)

    # recover
    X_recover=U@Z+X_mean
    show_reconstruction(X,X_recover,10,H,W)

    # accuracy
    filepath=os.path.join('Yale_Face_Database','Testing')
    X_test,y_test=imread(filepath,H,W)
    acc=performance(X_test,y_test,Z,y,U,X_mean,3)
    print('acc: {:.2f}%'.format(acc*100))