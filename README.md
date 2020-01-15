# NCTU-ML-class
2019 NCTU資工所 Machine Learning class  
teacher:洪瑞鴻、邱維辰  

## HW04
__logistic:__  
* 對於gradient descent與newton's method，都要調learning rate (不同task要用不同learning rate)  

__EM alogo:__ 
* log likelihood
  1. likelihood要用+的，用乘的會變成極小值（0.X的784次方）  
  2. normalized Posterior時，由於大家都是負數，會造成原本最大值在normalized後變最小值，故要用negative normalized改善。
  3. E-step向量化很重要，10分鐘->5秒
  4. for迴圈請注意cache miss
  5. 正確率約28\%
  
* likelihood
  1. likelihood用乘的
  2. E-step也可向量化
  3. prior中容易有幾個class的prob變為0
  3. 正確率約4x\%

## HW05
GP參考: http://krasserm.github.io/2018/03/19/gaussian-processes/  
libsvm參考: https://github.com/cjlin1/libsvm

## HW06
eigenvalue & eigenvector: https://drive.google.com/drive/folders/1vpMZ8n42cQJmU54vu38uNZ542-dA6MJb?usp=sharing  

## HW07
fisherface參考: https://www.bytefish.de/blog/fisherfaces/  

