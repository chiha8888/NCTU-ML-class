from libsvm.svmutil import *
from load import load_x,load_y

if __name__=='__main__':
    X_train=load_x('X_train.csv')
    y_train=load_y('Y_train.csv')
    X_test=load_x('X_test.csv')
    y_test=load_y('Y_test.csv')
    kernel_types={'linear':'-t 0','polynomial':'-t 1','radial basis function':'-t 2'}

    accuracy=[]
    for k,param in kernel_types.items():
        model=svm_train(y_train,X_train,'-q '+param)
        p_label,p_acc,p_vals=svm_predict(y_test,X_test,model,'-q')
        accuracy.append(p_acc[0])

    i=0
    for k,v in kernel_types.items():
        print('{} kernel accuracy: {:.2f}%'.format(k,accuracy[i]))
        i+=1