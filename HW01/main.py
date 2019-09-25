from rLSE import rlse
from newtonmethod import newtonmethod
import numpy as np
import matplotlib.pyplot as plt
import os

def show_fitting_line(parameters):
    parameters=parameters.reshape(-1)
    n=len(parameters)-1
    print('Fitting line: ',end='')
    #x^(n-1) ~ x^1
    for i in range(n,0,-1):
        print(parameters[i],'X^',i,' + ',end='')
    #x^0
    print(parameters[0])

def plot(x1,b,parameters_rlse,parameters_newton):
    #rlse
    plt.subplot(2,1,1)
    plt.title('rlse')
    plt.plot(x1,b,'ro')
    x1_min=min(x1)
    x1_max=max(x1)
    x=np.linspace(x1_min-1,x1_max+1,500)
    y=np.zeros(x.shape)
    for i in range(len(parameters_rlse)):
        y+=parameters_rlse[i]*np.power(x,i)
    plt.plot(x,y,'-k')
    #newton
    plt.subplot(2,1,2)
    plt.title('newton\'s method')
    plt.plot(x1, b, 'ro')
    y = np.zeros(x.shape)
    for i in range(len(parameters_newton)):
        y += parameters_newton[i] * np.power(x, i)
    plt.plot(x, y, '-k')
    plt.show()


''' A's dtype must be float!!! 
    I stuck for this wasting me 2hr...'''
''' only one feature:x1 '''
x1=[]
y=[]

path=input('path: ')
name=input('name: ')
filepath=os.path.join(path,name)
fp=open(filepath,'r')
line=fp.readline()
while line:
    a,b=line.split(',')
    x1.append(float(a))
    y.append(float(b))
    line=fp.readline()

x1=np.asarray(x1,dtype='float').reshape((-1,1))
b=np.asarray(y,dtype='float').reshape((-1,1))

while True:
    polynomial_basis_size=int(input('n: '))
    LAMBDA=int(input('lambda: '))
    print(polynomial_basis_size)
    # get A by polynomial basis size
    A=np.empty((len(x1),polynomial_basis_size))
    for j in range(polynomial_basis_size):
        A[:,j]=np.power(x1,j).reshape(-1)
    #print(A)
    # rLSE
    parameters_rlse,loss_rlse=rlse(A,LAMBDA,b)
    print('LSE:')
    show_fitting_line(parameters_rlse)
    print('Total error: ',loss_rlse)
    print()

    # Netwon's method
    parameters_newton,loss_newton=newtonmethod(A,b)
    print('Newton\'s Method:')
    show_fitting_line(parameters_newton)
    print('Total error: ',loss_newton)
    print()
    plot(x1.reshape(-1), b.reshape(-1), parameters_rlse.reshape(-1),parameters_newton.reshape(-1))




