import matplotlib.pyplot as plt
import numpy as np


A=np.asarray([[74.72,45.86,29.24,22.24,20.76, 20.32, 39.58, 78.9,  75.3 ],
 [84.24, 48.38, 27.9,  22.86, 20.92, 20.28, 26.62, 78.92, 75.7 ],
 [92.88, 49.54, 33.8 , 21.76, 20.52, 20.2 , 39.64, 78.86 ,75.48],
 [96.82, 54.94 ,37.4 , 25.48, 21.22 ,20.26, 39.86, 78.82, 75.28],
 [97.8 , 83.88, 62.96, 45.3 , 30.58, 24.5 , 21.82, 20.52, 69.38],
 [97.86, 85.12, 64.86, 44.36, 31.26, 25.28 ,22.12, 20.72, 69.02],
 [97.82, 84.78, 64.78, 44.72, 31.14, 25.38 ,22.02, 27.34,68.9 ],
 [97.7 , 85.12, 65.4 , 45.7 , 31.68, 25.18, 22.  , 27.42, 62.56],
 [97.78, 85.02, 65.48, 44.58, 31.  , 25.38, 22.68, 20.8 , 69.24]]
)
log2c=log2g=[-4,-3,-2,-1,0,1,2,3,4]

#plot
fig,ax=plt.subplots()
ax.matshow(A,cmap=plt.cm.Blues)
ax.set_xticklabels(['']+log2g)
ax.xaxis.set_label_position('top')
ax.set_yticklabels(['']+log2c)
for i in range(len(log2c)):
    for j in range(len(log2g)):
        ax.text(i,j,'{:.2f}'.format(A[j,i]),va='center',ha='center')
ax.set_xlabel('lg(G)')
ax.set_ylabel('lg(C)')
plt.show()