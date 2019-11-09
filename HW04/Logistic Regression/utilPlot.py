import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(A,b,b_predict):
    '''
    let class0 be positive, class1 be negative
    ----------
    | TP  FN |  <= confusion matrix by HW
    | FP  TN |
    ----------
    :param A: (2N,3) shape matrix
    :param b: (2N,1) shape matrix
    :param b_predict: (2N,1) shape matrix
    :return: (confusion_matix, points to be class0, points to be class1)
    '''
    doubleN=len(A)
    b_concate=np.hstack((b,b_predict))
    TP=FP=FN=TN=0
    for pair in b_concate:
        if pair[0]==pair[1]==1:
            TP+=1
        elif pair[0]==pair[1]==0:
            TN+=1
        elif pair[0]==1 and pair[1]==0:
            FP+=1
        else:
            FN+=1
    matrix=np.empty((2,2))
    matrix[0,0],matrix[0,1],matrix[1,0],matrix[1,1]=TP,FN,FP,TN

    C0_predict=[]
    C1_predict=[]
    for i in range(doubleN):
        if b_predict[i]==0:
            C0_predict.append(A[i,1:])
        else:
            C1_predict.append(A[i,1:])

    return (matrix,np.array(C0_predict),np.array(C1_predict))

def print_w(w):
    print('w:')
    print(w[0])
    print(w[1])
    print(w[2])

def print_confusion_matrix(matrix):
    print('Confusion Matrix:')
    print('               Predict cluster 1  Predict cluster 2')
    print('Is cluster 1        {:.0f}               {:.0f}       '.format(matrix[0,0],matrix[0,1]))
    print('Is cluster 2        {:.0f}               {:.0f}       '.format(matrix[1,0],matrix[1,1]))
    print()
    print('Sensitivity (Successfully predict cluster 1): {}'.format(matrix[0,0]/(matrix[0,0]+matrix[1,0])))
    print('Specificity (Successfully predict cluster 2): {}'.format(matrix[0,0]/(matrix[0,0]+matrix[0,1])))

def plot(C0,C1,title):
    plt.figure()
    plt.plot(C0[:,0],C0[:,1],'ro')
    plt.plot(C1[:,0],C1[:,1],'bo')
    plt.title(title)
    plt.show()



