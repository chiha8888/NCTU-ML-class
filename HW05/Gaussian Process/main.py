import numpy as np
import matplotlib.pyplot as plt
from util import *


X,y=load_data()
beta=5

#kernel
K=kernel(X,X,alpha=1,length_scale=1)+1/beta*np.identity(len(X))

# mean and variance in range[-60,60]
x_line=np.linspace(-60,60,num=500)
mean_predict,variance_predict=predict(x_line,X,y,K,beta,alpha=1,length_scale=1)
mean_predict=mean_predict.reshape(-1)
variance_predict=np.sqrt(np.diag(variance_predict))

#plot
plt.plot(X,y,'bo')
plt.plot(x_line,mean_predict,'k-')
plt.fill_between(x_line,mean_predict+2*variance_predict,mean_predict-2*variance_predict,facecolor='salmon')
plt.xlim(-60,60)
plt.show()


