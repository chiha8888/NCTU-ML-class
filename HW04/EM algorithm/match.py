import numpy as np
from scipy.optimize import linear_sum_assignment

def distance(a,b):
    '''
    :param a: (784)
    :param b: (784)
    :return: euclidean distance between a and b
    '''
    return np.linalg.norm(a-b)

def perfect_matching(ground_truth,estimate):
    '''
    matching GT_distribution to estimate_distribution by minimizing the sum of distance
    :param ground_truth: (10,784)
    :param estimate: (10,784)
    :return: (10)
    '''
    Cost=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            Cost[i,j]=distance(ground_truth[i],estimate[j])

    classes_order=hungarian_algo(Cost)

    return classes_order

def hungarian_algo(Cost):
    '''
    match GT to our estimate
    :param Cost: (10,10)
    :return: (10) column index
    '''
    row_idx,col_idx=linear_sum_assignment(Cost)
    return col_idx

