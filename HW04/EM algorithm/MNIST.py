import numpy as np
from keras.datasets import mnist

def load():
    (X_train,y_train),(_,_)=mnist.load_data()
    X_train=(X_train>=128).astype(int)
    X_train=X_train.reshape(len(X_train),-1)
    return (X_train,y_train)