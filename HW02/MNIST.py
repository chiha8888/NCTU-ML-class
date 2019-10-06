import numpy as np

def load():
    '''
    :return: (60000,784)&(60000)&(10000,784)&(10000) 4 ndarray
    '''
    '''
    from keras.datasets.mnist import load_data
    (train_x,train_y),(test_x,test_y)=load_data()
    return (train_x,train_y),(test_x,test_y)
    '''
    train_x_file=open('train-images.idx3-ubyte','rb')
    train_y_file=open('train-labels.idx1-ubyte','rb')
    test_x_file=open('t10k-images.idx3-ubyte','rb')
    test_y_file=open('t10k-labels.idx1-ubyte','rb')

    #train_x,train_y
    train_x_file.read(16)
    train_y_file.read(8)
    train_x=np.zeros((60000,28*28),dtype='uint8')
    train_y=np.zeros(60000,dtype='uint8')
    for i in range(60000):
        for j in range(28*28):
            train_x[i,j]=int.from_bytes(train_x_file.read(1),byteorder='big')
        train_y[i]=int.from_bytes(train_y_file.read(1),byteorder='big')

    #train_y,test_y
    test_x_file.read(16)
    test_y_file.read(8)
    test_x = np.zeros((10000, 28*28),dtype='uint8')
    test_y=np.zeros(10000,dtype='uint8')
    for i in range(10000):
        for j in range(28*28):
            test_x[i,j] = int.from_bytes(test_x_file.read(1), byteorder='big')
        test_y[i]=int.from_bytes(test_y_file.read(1),byteorder='big')

    return (train_x,train_y),(test_x,test_y)