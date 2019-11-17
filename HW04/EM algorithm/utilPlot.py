import numpy as np

def plot(Distribution,classes_order,threshold):
    '''
    plot each classes expected pattern
    :param Distribution: (10,784)
    :param classes_order: (10)
    :param threshold: value between 0.0~1.0
    :return:
    '''
    Pattern=np.asarray(Distribution>threshold,dtype='uint8')
    for i in range(10):
        print('class {}:'.format(i))
        plot_pattern(Pattern[classes_order[i]])
    return

def confusion_matrix(real,predict,classes_order):
    '''
    :param real: (60000)
    :param predict: (60000)
    :param classes_order: (10)
    :return:
    '''
    for i in range(10):
        c=classes_order[i]
        TP,FN,FP,TN=0,0,0,0
        for i in range(60000):
            if real[i]!=c and predict[i]!=c:
                TN+=1
            elif real[i]==c and predict[i]==c:
                TP+=1
            elif real[i]!=c and predict[i]==c:
                FP+=1
            else:
                FN+=1
        plot_confusion_matrix(c,TP,FN,FP,TN)

def print_error_rate(count,real,predict,classes_order):
    '''
    :param count: int
    :param real: (60000)
    :param predict: (60000)
    :param classes_order: (10)
    :return:
    '''
    print('Total iteration to converge: {}'.format(count))
    real_transform=np.zeros(60000)
    for i in range(60000):
        real_transform[i]=classes_order[real[i]]
    error=np.count_nonzero(real_transform-predict)
    print('Total error rate: {}'.format(error/60000))



def plot_confusion_matrix(c,TP,FN,FP,TN):
    print('------------------------------------------------------------')
    print()
    print('Confusion Matrix {}:'.format(c))
    print('\t\t\t  Predict number {} Predict not number {}'.format(c, c))
    print('Is number  \t{}\t\t{}\t\t\t\t{}'.format(c,TP,FN))
    print('Isn\'t number {}\t\t{}\t\t\t\t{}'.format(c,FP,TN))
    print()
    print('Sensitivity (Successfully predict number {}    ): {:.5f}'.format(c,TP/(TP+FN)))
    print('Specificity (Successfully predict not number {}): {:.5f}'.format(c,TN/(TN+FP)))
    print()

def plot_pattern(pattern):
    '''
    :param pattern: (784)
    :return:
    '''
    for i in range(28):
        for j in range(28):
            print(pattern[i*28+j],end=' ')
        print()
    print()
    print()
    return

