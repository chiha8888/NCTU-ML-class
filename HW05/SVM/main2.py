from load import load_x,load_y
from grid import grid_search
from plot import plot_confusion_matrix

if __name__=='__main__':
    X_train=load_x('X_train.csv')
    y_train=load_y('Y_train.csv')
    X_test=load_x('X_test.csv')
    y_test=load_y('Y_test.csv')

    log2c=log2g=[-4,-3,-2,-1,0,1,2,3,4]
    confusion_matrix=grid_search(log2c,log2g,X_train,y_train,X_test,y_test)

    plot_confusion_matrix(confusion_matrix,log2c,log2g)
