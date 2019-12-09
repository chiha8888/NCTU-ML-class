import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from util import *

def objective_function(X,y,beta):
    '''
    :param X:  (n) ndarray
    :param y:  (n) ndarray
    :param beta:
    :return:
    '''
    def objective(theta):
        K=kernel(X,X,alpha=theta[0],length_scale=theta[1])+(1/beta)*np.identity(len(X))
        L=np.linalg.cholesky(K)
        return 0.5*y.reshape(1,-1) @ np.linalg.inv(K) @ y.reshape(-1,1) + np.sum(np.log(np.diag(L))) + 0.5*len(X)*np.log(2*np.pi)

    return objective

X,y=load_data()
beta=5

objective_value=1e9
inits=[1e-2,1e-1,0,1e1,1e2]
for init_alpha in inits:
    for init_length_scale in inits:
        res=minimize(objective_function(X,y,beta),x0=[init_alpha,init_length_scale],bounds=((1e-5,1e5),(1e-5,1e5)))
        if res.fun<objective_value:
            objective_value = res.fun
            alpha_optimize,length_scale_optimize=res.x
print('alpha: ',alpha_optimize)
print('length_scale: ',length_scale_optimize)

#kernel
K=kernel(X,X,alpha=alpha_optimize,length_scale=length_scale_optimize)+1/beta*np.identity(len(X))

# mean and variance in range[-60,60]
x_line=np.linspace(-60,60,num=500)
mean_predict,variance_predict=predict(x_line,X,y,K,beta,alpha=alpha_optimize,length_scale=length_scale_optimize)
mean_predict=mean_predict.reshape(-1)
variance_predict=np.sqrt(np.abs(np.diag(variance_predict)))

#plot
plt.plot(X,y,'bo')
plt.plot(x_line,mean_predict,'k-')
plt.fill_between(x_line,mean_predict+2*variance_predict,mean_predict-2*variance_predict,facecolor='salmon')
plt.xlim(-60,60)
plt.show()
