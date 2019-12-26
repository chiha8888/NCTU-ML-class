import cv2
import os
from util import imread,precomputed_kernel,save_gif
from kmeans import kmeans


# set parameters
img_path='image1.png'
image_flat,HEIGHT,WIDTH=imread(img_path)
gamma_s=0.001
gamma_c=0.001
k=3  # k clusters
k_means_initType='k_means_plusplus'
gif_path=os.path.join('GIF','{}_{}Clusters_{}'.format(img_path.split('.')[0],k,'kernel k-means.gif'))

Gram=precomputed_kernel(image_flat,gamma_s,gamma_c)
belongings,segments=kmeans(Gram,k,HEIGHT,WIDTH,initType=k_means_initType)
save_gif(segments,gif_path)


cv2.waitKey(0)
cv2.destroyAllWindows()