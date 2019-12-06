import numpy as np
import matplotlib.pyplot as plt
from util import *


x,y=load_data()
beta=5

#kernel
k=get_kernel(x,beta)

# mean and variance in range[-60,60]
x_line=np.linspace(-60,60,num=500)
mean_predict=get_mus(x_line,x,y,k)
variance_predict=get_vars(x_line,x,k,beta)

#plot
plt.plot(x,y,'bo')
plt.plot(x_line,mean_predict,'k-')
plt.fill_between(x_line,mean_predict+2*variance_predict,mean_predict-2*variance_predict,facecolor='salmon')
plt.xlim(-60,60)
plt.show()


