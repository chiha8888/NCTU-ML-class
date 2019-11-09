import numpy as np
import matplotlib.pyplot as plt
from util import *
from utilPlot import *
from gradientDescent import run_gradient
from newtonMethod import run_newton

#input
d=int(input('default(0/1/2): '))
if d==0:
    N=int(input('N: '))
    mx1,my1=[int(x) for x in input('mx1與my1: ').split()]
    mx2,my2=[int(x) for x in input('mx2與my2: ').split()]
    vx1,vy1=[int(x) for x in input('vx1與vy1: ').split()]
    vx2,vy2=[int(x) for x in input('vx2與vy2: ').split()]
else:
    N,mx1,my1,mx2,my2,vx1,vy1,vx2,vy2=default(d)

#init w=[w0,w1,w2],A,b   sigmoid(Aw)=b
C0=sampling(mx1,my1,vx1,vy1,N)
C1=sampling(mx2,my2,vx2,vy2,N)
A=get_A(C0,C1)
b=get_b(N)

plot(C0,C1,'Ground truth')

#gradient descent
w=np.random.rand(3,1)
w=run_gradient(A,w,b,lr=0.001)

#print
print('Gradient descent:\n')
b_predict=predict(A,w)
matrix,C0_predict,C1_predict=confusion_matrix(A,b,b_predict)
print_w(w)
print_confusion_matrix(matrix)
plot(C0_predict,C1_predict,'Gradient descent')

#newton's method
w=np.random.rand(3,1)
w=run_newton(A,w,b,lr=0.001)


#print
print('\n----------------------------------------')
print('Newton\s method:')
b_predict=predict(A,w)
matrix,C0_predict,C1_predict=confusion_matrix(A,b,b_predict)
print_w(w)
print_confusion_matrix(matrix)
plot(C0_predict,C1_predict,'Newton\'s method')
