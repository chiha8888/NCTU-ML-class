import numpy as np
import cv2
import os
from util import imread,precomputed_kernel,plot_eigenvector,save_gif
from kmeans import kmeans
EPS=1e-9

# set parameters
img_path='image2.png'
image_flat,HEIGHT,WIDTH=imread(img_path)
gamma_s=0.001
gamma_c=0.001
k=3  # k clusters
k_means_initType='k_means_plusplus'
gif_path=os.path.join('GIF','{}_{}Clusters_{}'.format(img_path.split('.')[0],k,'normalized.gif'))

# similarity matrix
W=precomputed_kernel(image_flat,gamma_s,gamma_c)
# degree matrix
D=np.diag(np.sum(W,axis=1))
L=D-W
D_inverse_square_root=np.diag(1/np.diag(np.sqrt(D)))
L_sym=D_inverse_square_root@L@D_inverse_square_root

'''
eigenvalue,eigenvector=np.linalg.eig(L_sym)
np.save('{}_eigenvalue_{:.3f}_{:.3f}_normalized'.format(img_path.split('.')[0],gamma_s,gamma_c),eigenvalue)
np.save('{}_eigenvector_{:.3f}_{:.3f}_normalized'.format(img_path.split('.')[0],gamma_s,gamma_c),eigenvector)
'''

eigenvalue=np.load('{}_eigenvalue_{:.3f}_{:.3f}_normalized.npy'.format(img_path.split('.')[0],gamma_s,gamma_c))
eigenvector=np.load('{}_eigenvector_{:.3f}_{:.3f}_normalized.npy'.format(img_path.split('.')[0],gamma_s,gamma_c))
sort_index=np.argsort(eigenvalue)
# U:(n,k)
U=eigenvector[:,sort_index[1:1+k]]
# T:(n,k) each row with norm 1
sums=np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1)
T=U/sums

# k-means
belonging,segments=kmeans(T,k,HEIGHT,WIDTH,initType=k_means_initType)

save_gif(segments,gif_path)
if k==3:
    plot_eigenvector(U[:,0],U[:,1],U[:,2],belonging)

cv2.waitKey(0)
cv2.destroyAllWindows()