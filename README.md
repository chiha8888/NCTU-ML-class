# NCTU-ML-class
2019 NCTU資工所 Machine Learning class  
teacher:洪瑞鴻、邱維辰  

### HW04
__logistic:__  
1. 對於gradient descent與newton's method，都要調learning rate (不同task要用不同learning rate)  

__EM alogo:__  
1. likelihood要用+的，用乘的會變成極小值（0.X的784次方）  
2. normalized Posterior時，由於大家都是負數，會造成原本最大值在normalized後變最小值，故要用negative normalized改善。
3. E-step向量化很重要，10分鐘->5秒
4. for迴圈請注意cache miss
