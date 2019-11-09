import numpy as np
from gradientDescent import run_gradient

eps=1e-2

def run_newton(A,w,b,lr=0.01):
    N=len(A)
    D=np.zeros((N,N))
    for i in range(N):
        D[i,i]=np.exp(-A[i]@w)/np.power(1+np.exp(-A[i]@w),2)
    H=A.T@D@A
    try:
        H_inv=np.linalg.inv(H)
    except np.linalg.LinAlgError as error:
        print(str(error))
        print('Hessian matrix non invertible, switch to Gradient descent')
        return run_gradient(A,w,b)

    g=100
    while np.sqrt(np.sum(g**2))>eps:
        g=H_inv@A.T@(b-1/(1+np.exp(-A@w)))
        w=w+lr*g
        #print('w={}'.format(w.reshape(1,-1)))
        #print('g={}'.format(g.reshape(1,-1)))

    return w

