from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def imread(path,H,W):
    '''
    :param path:
    :param H:
    :param W:
    :return:  (W*H, # pics) ndarray , (# pics) ndarray
    '''
    pics=os.listdir(path)
    images=np.zeros((W*H,len(pics)))
    labels=np.zeros(len(pics)).astype('uint8')
    for pic,i in zip(pics,np.arange(len(pics))):
        labels[i]=int(pic.split('.')[0][7:9])-1
        image=np.asarray(Image.open(os.path.join(path,pic)).resize((W,H),Image.ANTIALIAS)).flatten()
        images[:,i]=image

    return images,labels

def show_eigenface(X,num,H,W):
    '''
    :param X: (H*W, low-dim) ndarray
    :param num: # of showing faces
    :param H:
    :param W:
    :return: 
    '''
    n=int(num**0.5)
    for i in range(num):
        plt.subplot(n,n,i+1)
        plt.imshow(X[:,i].reshape(H,W),cmap='gray')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

def show_reconstruction(X,X_recover,num,H,W):
    '''
    :param X:  (H*W,# person picture=135) ndarray
    :param X_recover:  (H*W# person picture=135) ndarray
    :param num:  # of showing faces
    :param H:
    :param W:
    :return:
    '''
    randint=np.random.choice(X.shape[1],num)
    for i in range(num):
        plt.subplot(2,num,i+1)
        plt.imshow(X[:,randint[i]].reshape(H,W),cmap='gray')
        plt.subplot(2,num,i+1+num)
        plt.imshow(X_recover[:,randint[i]].reshape(H,W),cmap='gray')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

def performance(X_test,y_test,Z_train,y_train,U,X_mean=None,k=3):
    '''
    using k-nn to predict X_test's label
    :param X_test:  (H*W, # pics) ndarray
    :param y_test:   (# pics) ndarray
    :param Z_train:  (low-dim, #pics) ndarray
    :param y_train:  (# pics) ndarray
    :param U: Transform matrix
    :param X_mean:  using when estimate eigenface
    :param k: k of k-nn
    :return:
    '''
    if X_mean is None:
        X_mean=np.zeros((X_test.shape[0],1))

    # reduce dim (projection)
    Z_test=U.T@(X_test-X_mean)

    # k-nn
    predicted_y=np.zeros(Z_test.shape[1])
    for i in range(Z_test.shape[1]):
        distance=np.zeros(Z_train.shape[1])
        for j in range(Z_train.shape[1]):
            distance[j]=np.sum(np.square(Z_test[:,i]-Z_train[:,j]))
        sort_index=np.argsort(distance)
        nearest_neighbors=y_train[sort_index[:k]]
        unique, counts = np.unique(nearest_neighbors, return_counts=True)
        nearest_neighbors=[k for k,v in sorted(dict(zip(unique, counts)).items(), key=lambda item: -item[1])]
        predicted_y[i]=nearest_neighbors[0]

    acc=np.count_nonzero((y_test-predicted_y)==0)/len(y_test)
    return acc
