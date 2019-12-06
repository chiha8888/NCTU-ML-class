import numpy as np

def load_data(path='input.data'):
    x=[]
    y=[]
    f = open(path, 'r')
    for line in f.readlines():
        datapoint = line.split(' ')
        x.append(float(datapoint[0]))
        y.append(float(datapoint[1]))
    f.close()
    x=np.asarray(x)
    y=np.asarray(y)
    return x,y

def get_kernel(x,beta):
    '''
    :param x: x_points
    :param beta: a noise scalar
    return: (n,n) kernel ndarray computed from datapoints
    '''
    n=len(x)
    kernel=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            kernel[i,j]=kernel[j,i]=kernel_function(x[i],x[j])
    kernel=kernel+(1/beta)*np.identity(n)
    return kernel

def kernel_function(a,b,alpha=1,length_scale=1):
    '''
    using rational quadratic kernel: k(x_i, x_j) = (1 + (x_i-x_j)^2 / (2*alpha * length_scale^2))^-alpha
    :return: a scalar
    '''
    cov = np.power((1+(np.power(a-b,2)/(2*alpha*length_scale**2))),-alpha)
    return cov

def get_mus(x_line,x,y,kernel):
    '''
    :param x_line: sampling by linspace(-60,60)
    :param x: datapoints x
    :param y: datapoints y
    :param kernel: (n,n) matrix
    return: (len(x_line)) array
    '''
    n=len(x_line)
    means=np.zeros(n)
    for i in range(n):
        k_x_xstar=kernel_function(x_line[i],x)
        means[i]=k_x_xstar @ np.linalg.inv(kernel) @ y.reshape(-1,1)
    return means

def get_vars(x_line,x,kernel,beta):
    '''
    :param x_line: sampling by linspace(-60,60)
    :param x: datapoints x
    :param kernel: (n,n) matrix
    :param beta: noise scalar
    :return: (len(x_line)) array
    '''
    n=len(x_line)
    variances=np.zeros(n)
    for i in range(n):
        k_x_xstar=kernel_function(x_line[i],x)
        variances[i]=(kernel_function(x_line[i],x_line[i]) + 1/beta) + k_x_xstar @ np.linalg.inv(kernel) @ k_x_xstar.T
    return variances
