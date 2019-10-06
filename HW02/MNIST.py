from keras.datasets.mnist import load_data

def load():
    (train_x,train_y),(test_x,test_y)=load_data()
    return (train_x,train_y),(test_x,test_y)