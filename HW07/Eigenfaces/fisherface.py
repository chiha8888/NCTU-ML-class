import numpy as np
import os
import matplotlib.pyplot as plt
from util import imread,show_eigenface,show_reconstruction,performance
from pca import pca
from lda import lda

if __name__=='__main__':
    filepath=os.path.join('Yale_Face_Database','Training')
    H,W=231,195
    X,y=imread(filepath,H,W)

    eigenvalues_pca,eigenvectors_pca,X_mean=pca(X,num_dim=31)
    X_pca=eigenvectors_pca.T@(X-X_mean)
    eigenvalues_lda,eigenvectors_lda=lda(X_pca,y)

    # Transform matrix
    U=eigenvectors_pca@eigenvectors_lda
    print('U shape: {}'.format(U.shape))

    # show top 25 eigenface
    show_eigenface(U,25,H,W)

    # reduce dim (projection)
    Z=U.T@X

    # recover
    X_recover=U@Z+X_mean
    show_reconstruction(X,X_recover,10,H,W)

    # accuracy
    filepath = os.path.join('Yale_Face_Database', 'Testing')
    X_test, y_test = imread(filepath, H, W)
    acc = performance(X_test, y_test, Z, y, U, X_mean, 5)
    print('acc: {:.2f}%'.format(acc * 100))
